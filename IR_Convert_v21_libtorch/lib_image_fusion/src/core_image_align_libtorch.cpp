#include <core_image_align_libtorch.h>
#include "util_timer.h"
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <experimental/filesystem>

namespace core
{
  ImageAlign::ImageAlign(Param param) : param_(std::move(param))
  {
    torch::manual_seed(1);
    torch::autograd::GradMode::set_enabled(false);

    if (param_.device.compare("cuda") == 0 && torch::cuda::is_available())
    {
      torch::Device cuda(torch::kCUDA);
      device = cuda;
    }

    net = torch::jit::load(param_.model_path);
    net.eval();
    net.to(device);

    if (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0)
      net.to(torch::kHalf);

    printf("Model initialization completed\n");

    // 執行 warmup，不論是 CPU 或 CUDA
    warm_up();
  }

  // warm up
  void ImageAlign::warm_up()
  {
    printf("Warm up...\n");

    cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;
    cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
    {
      cv::Mat H;
      std::vector<cv::Point2i> eo_mkpts, ir_mkpts;
      pred(eo, ir, eo_mkpts, ir_mkpts);
    }

    const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
    const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    printf("Warm up done in %.2f s\n", period);
  }

  // prediction - MODIFIED: 改善精度和資料處理，符合Python版本
  void ImageAlign::pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, const std::string& filename)
  {
    if (eo.channels() != 1 || ir.channels() != 1)
      throw std::runtime_error("ImageAlign::pred: eo and ir must be single channel images");

    // OPTIMAL: 最簡潔和準確的FP16/FP32處理方案
    // 策略：只在模型推理時使用FP16，輸入處理和輸出都用FP32以保持精度
    
    // 1. 直接創建FP32 tensor並正規化（只要1行！）
    torch::Tensor eo_tensor = torch::from_blob(eo.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).to(device).to(torch::kFloat32) / 255.0f;
    torch::Tensor ir_tensor = torch::from_blob(ir.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).to(device).to(torch::kFloat32) / 255.0f;
    
    // 2. 只在模型推理時才轉換為FP16（如果需要）
    // #TODO:因為pytorch在推論會使用到混合精度所以才會比較慢，libtorch市全部都是fp16所以才比較快，還有因為全部都是fp16所以誤差容易放大
    // #TODO:libtorch onnx trt 全部重測試 + warm up
    

    bool use_fp16 = (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0);
    if (use_fp16) {
      eo_tensor = eo_tensor.to(torch::kHalf);
      ir_tensor = ir_tensor.to(torch::kHalf);
    }

    // 計時 - 模型推論開始
    auto model_inference_start = std::chrono::high_resolution_clock::now();

    // // // 確保CUDA操作完成後再開始計時推論
    // if (param_.device.compare("cuda") == 0) {
    //   torch::cuda::synchronize();
    // }

    // run the model
    torch::IValue pred = net.forward({eo_tensor, ir_tensor});
    
    // // 確保推論完成
    // if (param_.device.compare("cuda") == 0) {
    //   torch::cuda::synchronize();
    // }
    
    // 計時 - 模型推論結束
    auto model_inference_end = std::chrono::high_resolution_clock::now();
    double model_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(model_inference_end - model_inference_start).count() / 1000000.0; // 轉換為秒
    torch::jit::Stack pred_ = pred.toTuple()->elements();

    // get mkpts from the model output
    torch::Tensor eo_mkpts = pred_[0].toTensor().to(torch::kFloat32); // 確保輸出是FP32
    torch::Tensor ir_mkpts = pred_[1].toTensor().to(torch::kFloat32);
    int leng=pred_[2].toInt(); // 獲取特徵點數量

    // clean up eo_pts and ir_pts
    eo_pts.clear();
    ir_pts.clear();

    for (int i = 0; i <leng; i++)
    {
      // 使用round而非直接轉換，提高精度
      float eo_x = eo_mkpts[i][0].item<float>();
      float eo_y = eo_mkpts[i][1].item<float>();
      float ir_x = ir_mkpts[i][0].item<float>();
      float ir_y = ir_mkpts[i][1].item<float>();
      
      // eo_pts.push_back(cv::Point2i(static_cast<int>(std::round(eo_x)), static_cast<int>(std::round(eo_y))));
      // ir_pts.push_back(cv::Point2i(static_cast<int>(std::round(ir_x)), static_cast<int>(std::round(ir_y))));
      eo_pts.push_back(cv::Point2i(static_cast<int>(eo_x), static_cast<int>(eo_y)));
      ir_pts.push_back(cv::Point2i(static_cast<int>(ir_x), static_cast<int>(ir_y)));
    }
    
    // 寫入CSV檔案 - 只記錄模型推論時間
    writeTimingToCSV("Model_Inference", model_inference_time, leng, filename);
    
    printf("Model inference time: %.6f s\n", model_inference_time);

    // DEBUG: 輸出特徵點數量
    std::cout << "  - Model extracted " << eo_pts.size() << " feature point pairs" << std::endl;
  }

  // align with last H - MODIFIED: 簡化過濾邏輯，使用Python風格的直接輸出
  void ImageAlign::align(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, cv::Mat &H, const std::string& filename)
  {
    // predict keypoints
    pred(eo, ir, eo_pts, ir_pts, filename);

    // CORRECTED: 與Python代碼完全一致的特徵點縮放條件檢查
    if (std::abs(param_.out_width_scale - 1.0) > 1e-6 || std::abs(param_.out_height_scale - 1.0) > 1e-6 || param_.bias_x > 0 || param_.bias_y > 0)
    {
      for (cv::Point2i &i : eo_pts)
      {
        i.x = i.x * param_.out_width_scale  ;
        i.y = i.y * param_.out_height_scale ;
      }
      for (cv::Point2i &i : ir_pts)
      {
        i.x = i.x * param_.out_width_scale  ;
        i.y = i.y * param_.out_height_scale ;
      }
      
      std::cout << "  - Feature point scaling applied: pred(" << param_.pred_width << "x" << param_.pred_height 
                << ") -> out(" << param_.out_width_scale * param_.pred_width << "x" << param_.out_height_scale * param_.pred_height 
                << "), scale=(" << param_.out_width_scale << ", " << param_.out_height_scale 
                << "), bias=(" << param_.bias_x << ", " << param_.bias_y << ")" << std::endl;
    }
    else
    {
      std::cout << "  - No feature point scaling needed: scale=(" << param_.out_width_scale << ", " << param_.out_height_scale 
                << "), bias=(" << param_.bias_x << ", " << param_.bias_y << ")" << std::endl;
    }
    
    // 輸出最終特徵點數量
    std::cout << "  - Final feature points after coordinate adjustment: " << eo_pts.size() << std::endl;
  }

  // 寫入計時資料到CSV檔案
  void ImageAlign::writeTimingToCSV(const std::string& operation, double time_ms, int leng, const std::string& filename)
  {
    std::string csv_filename = "./itiming_log.csv";
    bool file_exists = std::experimental::filesystem::exists(csv_filename);
    
    std::ofstream csv_file(csv_filename, std::ios::app);
    
    if (!file_exists) {
      // 寫入CSV標頭
      csv_file << "Filename,Operation,Time_s,Mode,keypoints\n";
    }
    
    // 使用檔案名稱代替時間戳，如果沒有提供檔案名稱則使用時間戳
    std::string identifier;
    if (!filename.empty()) {
      identifier = filename;
    } else {
      std::cout<<"warm up"<<std::endl;
      return;
    }
    
    // 寫入資料
    csv_file << identifier << "," 
             << operation << "," 
             << std::fixed << std::setprecision(6) << time_ms << ","
             << param_.mode << ","
             << leng << "\n";
    
    csv_file.close();
  }
} /* namespace core */

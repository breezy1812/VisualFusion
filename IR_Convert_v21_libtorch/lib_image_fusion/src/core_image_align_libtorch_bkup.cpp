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

    // 智能 warmup: 執行 warmup 來初始化 CUDA kernels，但不保留內部狀態
    if (param_.device.compare("cuda") == 0) {
      printf("Performing smart warmup to initialize CUDA kernels...\n");
      smart_warmup();
    }
  }

  // warm up
  // void ImageAlign::warm_up()
  // {
  //   printf("Warm up...\n");

  //   cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;
  //   cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;

  //   const auto t0 = std::chrono::high_resolution_clock::now();
  //   for (int i = 0; i < 5; i++)
  //   {
  //     cv::Mat H;
  //     std::vector<cv::Point2i> eo_mkpts, ir_mkpts;
  //     pred(eo, ir, eo_mkpts, ir_mkpts);
  //   }

  //   const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
  //   const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

  //   printf("Warm up done in %.2f s\n", period);
  // }

  // smart warmup: 初始化 CUDA kernels 但保持精度
  void ImageAlign::smart_warmup()
  {
    printf("Smart warmup for CUDA kernel initialization...\n");

    cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
    cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;

    const auto t0 = std::chrono::high_resolution_clock::now();
    
    // 只執行一次推理來初始化 CUDA kernels
    torch::Tensor eo_tensor = torch::from_blob(eo.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).clone().to(device).to(torch::kFloat32) / 255.0f;
    torch::Tensor ir_tensor = torch::from_blob(ir.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).clone().to(device).to(torch::kFloat32) / 255.0f;
    
    bool use_fp16 = (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0);
    if (use_fp16) {
      eo_tensor = eo_tensor.to(torch::kHalf);
      ir_tensor = ir_tensor.to(torch::kHalf);
    }
    
    // 執行一次推理（結果被丟棄）
    torch::IValue pred = net.forward({eo_tensor, ir_tensor});
    
    // 重新載入模型以清除內部狀態，保持第一次推理的精度
    printf("Reloading model to maintain first-inference precision...\n");
    net = torch::jit::load(param_.model_path);
    net.eval();
    net.to(device);
    if (use_fp16) {
      net.to(torch::kHalf);
    }

    const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
    const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    printf("Smart warmup completed in %.2f s\n", period);
  }

  // prediction - MODIFIED: 改善精度和資料處理，符合Python版本
  void ImageAlign::pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, const std::string& filename)
  {
    if (eo.channels() != 1 || ir.channels() != 1)
      throw std::runtime_error("ImageAlign::pred: eo and ir must be single channel images");

    cv::Mat eo_float, ir_float;
    eo.convertTo(eo_float, CV_32F, 1.0 / 255.0, 0.0);
    ir.convertTo(ir_float, CV_32F, 1.0 / 255.0, 0.0);

      // 1. 直接創建FP32 tensor並正規化（使用調整後的圖像）
    // torch::Tensor eo_tensor = torch::from_blob(eo.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).clone().to(device).to(torch::kFloat32) / 255.0f;
    // torch::Tensor ir_tensor = torch::from_blob(ir.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).clone().to(device).to(torch::kFloat32) / 255.0f;
    // 使用 cv::Mat 轉換為 float (CV_32F)，然後直接用 from_blob 建立 FP32 tensor，並正規化到 [0,1]
    torch::Tensor eo_tensor = torch::from_blob(eo_float.ptr<float>(), {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).clone().to(device);
    torch::Tensor ir_tensor = torch::from_blob(ir_float.ptr<float>(), {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).clone().to(device);

    // bool use_fp16 = (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0);
    // if (use_fp16) {
    //   eo_tensor = eo_tensor.to(torch::kHalf);
    //   ir_tensor = ir_tensor.to(torch::kHalf);
    // }
    // //check input data, write it in csv (修正FP16精度問題)
    // if(!filename.empty()){
    //   std::string input_csv_filename = "/circ330/forgithub/VisualFusion_libtorch/IR_Convert_v21_libtorch/output/input_data_"+filename + ".csv";

    //   std::ofstream ofs(input_csv_filename);
    //   ofs << std::fixed << std::setprecision(20); // 控制小數點後20位
    //   if (!ofs) {
    //     throw std::runtime_error("Failed to open CSV file for writing");
    //   }

      
    //   for (int i = 0; i < eo_tensor.size(2); ++i) {
    //     for (int j = 0; j < eo_tensor.size(3); ++j) {
    //       ofs << eo_tensor[0][0][i][j].item<float>() << "," << ir_tensor[0][0][i][j].item<float>() << "\n";
    //     }
    //   }
    // }
    
    // 計時 - 模型推論開始
    auto model_inference_start = std::chrono::high_resolution_clock::now();

    // run the model

    torch::IValue pred = net.forward({eo_tensor, ir_tensor});
    // 獲取指定的 BatchNorm 模組並提取 running_mean 和 running_var
    // std::cout << "Extracting BatchNorm statistics..." << std::endl;
    // 列出所有模組名稱以便調試
    // auto named_modules = net.named_modules();
    // for (const auto& module : named_modules) {
    //   if (module.name.find("backbone.reg0.feat_trans.bn") == std::string::npos) {
    //     continue;
    //   }

    //   std::cout << "Found BatchNorm module: " << module.name << std::endl;
    //   // torch::Tensor running_mean = module.value.attr("running_mean").toTensor();
    //   // torch::Tensor running_var = module.value.attr("running_var").toTensor();
    //   double eps = module.value.attr("eps").toDouble();
    //   std::cout << "  - eps: " << std::fixed << std::setprecision(20) << eps << std::endl;

    //   // int len = eps.size(0);
    //   // for (int i = 0; i < len; ++i) {
    //   //   // std::cout << running_mean[i].item<float>() << ", " << running_var[i].item<float>() << std::endl;
    //   // }
    // }
    
    // named_modules();
    // 計時 - 模型推論結束
    auto model_inference_end = std::chrono::high_resolution_clock::now();
    double model_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(model_inference_end - model_inference_start).count() / 1000000.0; // 轉換為秒
    torch::jit::Stack pred_ = pred.toTuple()->elements();

    // // 獲取 bias 和 weight (現在模型輸出這兩個值)
    // torch::Tensor bias_tensor = pred_[0].toTensor();
    // torch::Tensor weight_tensor = pred_[1].toTensor();
    // // 寫入 bias_weight.csv
    // if(!filename.empty()){
    //   std::string bias_weight_csv_filename = "/circ330/forgithub/VisualFusion_libtorch/IR_Convert_v21_libtorch/output/bias_weight_data_"+filename + ".csv";
    //   std::ofstream bias_weight_csv(bias_weight_csv_filename);
    //   if (bias_weight_csv.is_open()) {
    //     std::cout << "Saving bias and weight data to CSV: " << bias_weight_csv_filename << std::endl;
        
    //     // 轉到 CPU 並 flatten
    //     torch::Tensor bias_flat = bias_tensor.detach().cpu().flatten();
    //     torch::Tensor weight_flat = weight_tensor.detach().cpu().flatten();
        
    //     int bias_size = bias_flat.size(0);
    //     int weight_size = weight_flat.size(0);
    //     int max_size = std::max(bias_size, weight_size);
        
    //     // bias_weight_csv << "bias,weight\n"; // 標題行
    //     bias_weight_csv << std::fixed << std::setprecision(20); // 高精度輸出
        
    //     for (int i = 0; i < max_size; ++i) {
    //       float bias_val = (i < bias_size) ? bias_flat[i].item<float>() : 0.0f;
    //       float weight_val = (i < weight_size) ? weight_flat[i].item<float>() : 0.0f;
    //       bias_weight_csv << bias_val << "," << weight_val << "\n";
    //     }
        
    //     bias_weight_csv.close();
    //     std::cout << "Bias shape: [" << bias_tensor.sizes() << "]" << std::endl;
    //     std::cout << "Weight shape: [" << weight_tensor.sizes() << "]" << std::endl;
    //   }
    // }

    // 寫入 feat_channels.csv
    
    torch::Tensor feat_vi = pred_[0].toTensor();
    torch::Tensor feat_ir = pred_[1].toTensor();
    if(!filename.empty()){
      std::string feat_csv_filename = "/circ330/forgithub/VisualFusion_libtorch/IR_Convert_v21_libtorch/output/feat_data_"+filename + ".csv";
      std::ofstream feat_csv(feat_csv_filename);
      if (feat_csv.is_open()) {
        // 取得 input tensor 的內容 (eo_tensor, ir_tensor)，先轉到 CPU 並 flatten
        std::cout<<"debug: Saving feature channels to CSV: " << feat_csv_filename << std::endl;
        torch::Tensor eo_flat = feat_vi.detach().cpu().flatten();
        torch::Tensor ir_flat = feat_ir.detach().cpu().flatten();
        int n = eo_flat.size(0);
        feat_csv << std::fixed << std::setprecision(20); // 控制小數點後8位
        for (int i = 0; i < n; ++i) {
          float vi_val = eo_flat[i].item<float>();
          float ir_val = ir_flat[i].item<float>();
          feat_csv << vi_val << "," << ir_val << "\n";
        }    // DEBUG1: 輸出 feat_vi 和 feat_ir 的 channel=[:] 資料
        
        
        // torch::Tensor feat_sa_vi = feat_sa_vi.detach().cpu().flatten();
        // int n = feat_sa_vi.size(0);
        // feat_csv << std::fixed << std::setprecision(20); // 控制小數點後8位
        // for (int i = 0; i < n; ++i) {
        //   float feat_sa_viv = feat_sa_vi[i].item<float>();
        //   feat_csv << feat_sa_viv <<  "\n";
        // }    // DEBUG1: 輸出 feat_vi 和 feat_ir 的 channel=[:] 資料
        feat_csv.close();
      }
    }

    // 注意：現在模型輸出bias和weight，不再輸出特徵點
    // 如果需要特徵點功能，請修改模型使其輸出特徵點
    
    // clean up eo_pts and ir_pts
    eo_pts.clear();
    ir_pts.clear();
    
    // 由於模型現在輸出bias和weight而不是特徵點，我們不能提取特徵點
    // 如果需要特徵點功能，需要修改模型或使用不同的處理方式
    std::cout << "Model now outputs bias and weight instead of keypoints." << std::endl;
    std::cout << "No keypoints extracted." << std::endl;
        
      //   // 寫入特徵點座標
      //     keypoints_csv_file << "(" << std::fixed << std::setprecision(1) 
    // 由於現在模型輸出bias和weight，不再有特徵點相關的處理
    std::cout << "Model now outputs bias and weight. No keypoints processing available." << std::endl;
    
    // 寫入CSV檔案 - 只記錄模型推論時間
    writeTimingToCSV("Model_Inference", model_inference_time, 0, filename); // leng設為0因為沒有特徵點
    
    printf("Model inference time: %.6f s\n", model_inference_time);

    // DEBUG: 輸出特徵點數量（現在為0）
    std::cout << "  - Model extracted 0 feature point pairs (bias/weight mode)" << std::endl;
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

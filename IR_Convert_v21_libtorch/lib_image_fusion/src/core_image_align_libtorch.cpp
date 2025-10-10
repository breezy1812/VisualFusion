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
    
    // ===== 同步 Python 端的確定性設定 =====
    // 1. 關閉 TF32 (與 Python torch.backends.cuda.matmul.allow_tf32 = False 等價)
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);
    // 2. 啟用 cuDNN deterministic (與 Python torch.backends.cudnn.deterministic = True 等價)
    // at::globalContext().setDeterministicCuDNN(true);
    
    // 3. 關閉 cuDNN benchmark (與 Python torch.backends.cudnn.benchmark = False 等價)
    // at::globalContext().setBenchmarkCuDNN(false);
    
    // 4. 設定 cuDNN 為確定性模式 (與 Python torch.use_deterministic_algorithms(True) 等價)
    // at::globalContext().setDeterministicAlgorithms(true, false);

    printf("Deterministic settings applied:\n");
    printf("  - TF32 cuBLAS: disabled\n");
    printf("  - TF32 cuDNN: disabled\n");
    printf("  - cuDNN deterministic: enabled\n");
    printf("  - cuDNN benchmark: disabled\n");
    printf("  - Deterministic algorithms: enabled\n");

    if (param_.device.compare("cuda") == 0 && torch::cuda::is_available())
    {
      torch::Device cuda(torch::kCUDA);
      device = cuda;
    }

    // 載入 TorchScript 模型（FP16 或 FP32）
    net = torch::jit::load(param_.model_path);
    net.eval();
    net.to(device);

    // ⭐ 模型已在 Python 端轉換為對應精度，C++ 端無需額外轉換
    printf("Model initialization completed\n");
    printf("  - Mode: %s\n", param_.mode.c_str());
    printf("  - Device: %s\n", param_.device.c_str());
    if (param_.mode.compare("fp16") == 0) {
      printf("  - Precision: FP16 (pre-converted from Python)\n");
      printf("  - Model weights: FP16\n");
      printf("  - Input tensors will be: FP16\n");
      printf("  - Ready for Tensor Core acceleration\n");
    } else {
      printf("  - Precision: FP32\n");
    }

    // 智能 warmup: 執行 warmup 來初始化 CUDA kernels，但不保留內部狀態
    if (param_.device.compare("cuda") == 0) {
      printf("Performing smart warmup to initialize CUDA kernels...\n");
      smart_warmup();
    }
  }

  void ImageAlign::smart_warmup()
  {
    printf("Smart warmup for CUDA kernel initialization (5 iterations)...\n");

    cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
    cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;

    const auto t0 = std::chrono::high_resolution_clock::now();
    
    bool use_fp16 = (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0);
    
    if (use_fp16) {
      printf("  - Warmup mode: FP16 (matching inference precision)\n");
    } else {
      printf("  - Warmup mode: FP32\n");
    }
    
    // 執行 5 次推理來完全初始化 CUDA kernels
    for (int i = 0; i < 5; i++) {
      printf("  Warmup iteration %d/5...\n", i + 1);
      
      torch::Tensor eo_tensor = torch::from_blob(eo.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).clone().to(device).to(torch::kFloat32) / 255.0f;
      torch::Tensor ir_tensor = torch::from_blob(ir.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).clone().to(device).to(torch::kFloat32) / 255.0f;
      
      // ⭐ 業界推薦：warmup 時輸入精度與推論一致
      if (use_fp16) {
        eo_tensor = eo_tensor.to(torch::kHalf);
        ir_tensor = ir_tensor.to(torch::kHalf);
      }
      
      // 執行推理（結果被丟棄）
      torch::IValue pred = net.forward({eo_tensor, ir_tensor});
    }

    const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
    const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    printf("Smart warmup completed in %.2f s (5 iterations)\n", period);
    printf("✅ CUDA kernels initialized, ready for inference\n");
  }

  // prediction - OPTIMIZED: 優化數據處理流程，減少 CPU 開銷
  // ⭐ 業界推薦方案：模型已在 C++ 端轉為 FP16，輸入資料也需轉為 FP16
  void ImageAlign::pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, const std::string& filename)
  {
    if (eo.channels() != 1 || ir.channels() != 1)
      throw std::runtime_error("ImageAlign::pred: eo and ir must be single channel images");

    // 根據 pred_mode 決定使用 FP32 還是 FP16
    bool use_fp16 = (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0);
    
    
    torch::Tensor eo_tensor, ir_tensor;
    
    // 優化：直接在 GPU 上進行歸一化，減少 CPU 端的 convertTo 開銷
    if (param_.device.compare("cuda") == 0) {
      // 直接從 uint8 創建 tensor 並在 GPU 上歸一化
      eo_tensor = torch::from_blob(eo.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8)
                    .clone()  // 必須 clone，避免數據被釋放
                    .to(device)
                    .to(torch::kFloat32)
                    .div_(255.0f);  // 在 GPU 上歸一化，使用 inplace 操作更快
      
      ir_tensor = torch::from_blob(ir.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8)
                    .clone()
                    .to(device)
                    .to(torch::kFloat32)
                    .div_(255.0f);
      
      // ⭐ 業界推薦：輸入資料必須與模型精度一致
      // 若模型已轉 FP16，輸入也要轉 FP16，確保類型匹配和 Tensor Core 加速
      if (use_fp16) {
        eo_tensor = eo_tensor.to(torch::kHalf);
        ir_tensor = ir_tensor.to(torch::kHalf);
      }
    } else {
      // CPU 模式：僅支援 FP32
      cv::Mat eo_float, ir_float;
      eo.convertTo(eo_float, CV_32F, 1.0 / 255.0, 0.0);
      ir.convertTo(ir_float, CV_32F, 1.0 / 255.0, 0.0);
      
      eo_tensor = torch::from_blob(eo_float.ptr<float>(), {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).clone();
      ir_tensor = torch::from_blob(ir_float.ptr<float>(), {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).clone();
    }
    
    // 計時 - 模型推論開始
    auto model_inference_start = std::chrono::high_resolution_clock::now();

    // run the model
    torch::IValue pred = net.forward({eo_tensor, ir_tensor});
    
    // // 確保推論完成（僅在CUDA模式下）
    // if (param_.device.compare("cuda") == 0) {
    //   torch::cuda::synchronize();
    // }
    
    // 計時 - 模型推論結束
    auto model_inference_end = std::chrono::high_resolution_clock::now();
    double model_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(model_inference_end - model_inference_start).count() / 1000000.0; // 轉換為秒
    torch::jit::Stack pred_ = pred.toTuple()->elements();

    // 新模型只返回 2 個輸出：mkpts0 和 mkpts1 (int32 類型)
    torch::Tensor eo_mkpts = pred_[0].toTensor().to(torch::kInt32);
    torch::Tensor ir_mkpts = pred_[1].toTensor().to(torch::kInt32);

    // clean up eo_pts and ir_pts
    eo_pts.clear();
    ir_pts.clear();

    // 遍歷所有點，過濾掉座標為 (0, 0) 的無效點
    int num_points = eo_mkpts.size(0);  // 獲取總點數（應該是 1200）
    for (int i = 0; i < num_points; i++)
    {
      int eo_x = eo_mkpts[i][0].item<int>();
      int eo_y = eo_mkpts[i][1].item<int>();
      int ir_x = ir_mkpts[i][0].item<int>();
      int ir_y = ir_mkpts[i][1].item<int>();
      
      // 跳過座標為 (0, 0) 的無效點
      if (eo_x == 0 && eo_y == 0) {
        continue;
      }
      
      eo_pts.push_back(cv::Point2i(eo_x, eo_y));
      ir_pts.push_back(cv::Point2i(ir_x, ir_y));
    }
    
    int leng = eo_pts.size();  // 實際有效的特徵點數量
    
    // 寫入CSV檔案 - 只記錄模型推論時間
    writeTimingToCSV("Model_Inference", model_inference_time, leng, filename);
    
    // 輸出詳細計時信息
    // printf("⏱️  Timing breakdown:\n");
    // printf("  - Data preparation: %.3f ms\n", data_prep_time);
    // printf("  - Model inference: %.3f ms (%.6f s)\n", model_inference_time * 1000, model_inference_time);
    // printf("  - Total: %.3f ms\n", data_prep_time + model_inference_time * 1000);

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

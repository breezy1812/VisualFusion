#include "../include/core_image_align_onnx.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstdlib>  // 為setenv函數添加
#include <onnxruntime_cxx_api.h>

// 嘗試包含CUDA頭文件，如果不存在則跳過
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace core {

class ImageAlignONNXImpl : public ImageAlignONNX {
private:
    Param param_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<Ort::AllocatedStringPtr> input_names_ptrs_;
    std::vector<Ort::AllocatedStringPtr> output_names_ptrs_;
    
    // CSV logging for inference times
    std::ofstream csv_file_;
    int inference_count_ = 0;
    std::string current_image_name_ = "";  // 新增：當前處理的圖片名稱
    
    // Smart warmup for ONNX: 初始化 provider 但不影響精度
    void smart_warmup_onnx() {
        std::cout << "Smart warmup for ONNX providers initialization..." << std::endl;
        
        cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
        cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
        
        const auto t0 = std::chrono::high_resolution_clock::now();
        
        // 只執行一次推理來初始化 ONNX providers
        std::vector<cv::Point2i> dummy_eo_mkpts, dummy_ir_mkpts;
        try {
            pred_cpu(eo, ir, dummy_eo_mkpts, dummy_ir_mkpts);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Smart warmup failed: " << e.what() << std::endl;
        }
        
        // 重新創建 session 以清除內部狀態，保持第一次推理的精度
        std::cout << "Recreating ONNX session to maintain first-inference precision..." << std::endl;
        session_.reset();
        session_ = std::make_unique<Ort::Session>(env_, param_.model_path.c_str(), session_options_);
        
        // 重新初始化 input/output names（這很重要！）
        input_names_.clear();
        output_names_.clear();
        input_names_ptrs_.clear();
        output_names_ptrs_.clear();
        
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input names
        size_t num_input_nodes = session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_ptrs_.push_back(std::move(input_name));
            input_names_.push_back(input_names_ptrs_.back().get());
        }
        
        // Get output names
        size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_ptrs_.push_back(std::move(output_name));
            output_names_.push_back(output_names_ptrs_.back().get());
        }
        
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        std::cout << "Smart warmup completed in " << dt.count() << " ms" << std::endl;
    }

public:
    ImageAlignONNXImpl(const Param& param) : param_(param), env_(ORT_LOGGING_LEVEL_WARNING, "ImageAlign") {
        // 設定隨機種子以確保與LibTorch C++代碼一致的推理結果
        std::cout << "debug: Setting deterministic seeds for ONNX inference..." << std::endl;
        std::srand(1);  // 與LibTorch C++一致：使用種子1
        srand(1);       // 確保所有C隨機函數都使用相同種子
        std::cout << "debug: Random seeds and environment configured for deterministic ONNX inference (seed=1, matching LibTorch C++)" << std::endl;
        
        // Initialize CSV file for logging inference times
        csv_file_.open("onnx_inference_times.csv", std::ios::app);
        if (!csv_file_.is_open()) {
            std::cerr << "Warning: Could not open CSV file for writing inference times" << std::endl;
        } else {
            // Write header if file is empty/new
            csv_file_.seekp(0, std::ios::end);
            if (csv_file_.tellp() == 0) {
                csv_file_ << "Image_Name,Inference_Time_Seconds,Features_Count" << std::endl;
            }
        }
        
        // 檢查模型文件是否存在
        if (!std::experimental::filesystem::exists(param_.model_path)) {
            std::cerr << "FATAL ERROR: Model file not found: " << param_.model_path << std::endl;
            throw std::runtime_error("ONNX model file not found: " + param_.model_path);
        }
        
        try {
            // ===== 🔒 ONNX Runtime 確定性與精度設定 =====
            std::cout << "debug: Configuring ONNX Runtime for FP32 precision and determinism..." << std::endl;
            
            // 強制單執行緒執行，避免並行導致的非確定性
            session_options_.SetIntraOpNumThreads(1);                    
            session_options_.SetInterOpNumThreads(1);                    
            
            // 設定確定性執行模式
            session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            
            // 禁用圖優化以確保確定性
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
            
            // 設定日志等級
            session_options_.SetLogSeverityLevel(3);
            
            // 額外的確定性設定（根據ONNX Runtime版本可能不可用）
            try {
                session_options_.DisableMemPattern();     // 禁用記憶體模式優化
            } catch (...) {
                std::cout << "debug: DisableMemPattern not available in this ONNX Runtime version" << std::endl;
            }
            
            try {
                session_options_.DisableCpuMemArena();    // 禁用CPU記憶體池
            } catch (...) {
                std::cout << "debug: DisableCpuMemArena not available in this ONNX Runtime version" << std::endl;
            }
            
            // Check if CUDA is requested and available
            std::cout << "debug: Attempting to initialize ONNX Runtime with device: " << param_.device << std::endl;
            
            bool use_cuda = false;
            if (param_.device == "cuda") {  // 使用CUDA
                std::cout << "debug: Adding CUDA execution provider with FP32 and deterministic settings..." << std::endl;
                
                // ===== 🚫 禁用 TF32，強制使用 FP32 =====
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                
                // 設定確定性算法選擇
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;  
                cuda_options.do_copy_in_default_stream = 1;    // 強制同步複製
                
                // 🔑 關鍵設定：禁用 TF32，強制使用 FP32
                // 透過環境變數禁用 TF32（這是最可靠的方法）
                setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
                std::cout << "✅ Set NVIDIA_TF32_OVERRIDE=0 to disable TF32" << std::endl;
                
                session_options_.AppendExecutionProvider_CUDA(cuda_options);
                use_cuda = true;
                std::cout << "✅ CUDA execution provider added with FP32 precision (TF32 disabled)" << std::endl;
            } else {
                std::cout << "debug: Using CPU execution provider" << std::endl;
            }
            
            session_ = std::make_unique<Ort::Session>(env_, param_.model_path.c_str(), session_options_);
            
            std::cout << "✅ ONNX Runtime session created with:" << std::endl;
            std::cout << "  - FP32 precision (TF32 disabled)" << std::endl;
            std::cout << "  - Deterministic mode enabled" << std::endl;
            std::cout << "  - Single-threaded execution" << std::endl;
            
            // Use CPU memory allocator for input preparation
            memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
            
            // Get input/output names
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Get input names
            size_t num_input_nodes = session_->GetInputCount();
            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = session_->GetInputNameAllocated(i, allocator);
                input_names_ptrs_.push_back(std::move(input_name));
                input_names_.push_back(input_names_ptrs_.back().get());
            }
            
            // Get output names
            size_t num_output_nodes = session_->GetOutputCount();
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = session_->GetOutputNameAllocated(i, allocator);
                output_names_ptrs_.push_back(std::move(output_name));
                output_names_.push_back(output_names_ptrs_.back().get());
            }
            
            std::cout << "Successfully loaded ONNX model with " << num_input_nodes << " inputs and " << num_output_nodes << " outputs" << std::endl;
            
            // 智能 warmup (可選)
            if (param_.device.compare("cuda") == 0) {
                std::cout << "ONNX CUDA model initialized without warmup to maintain first-inference precision" << std::endl;
            } else {
                std::cout << "ONNX CPU model initialized without warmup to maintain first-inference precision" << std::endl;
            }
            
        } catch (const Ort::Exception& e) {
            std::cerr << "FATAL ERROR: Failed to load ONNX model: " << e.what() << std::endl;
            throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
        }
    }
    
    // Destructor
    ~ImageAlignONNXImpl() {
        if (csv_file_.is_open()) {
            csv_file_.close();
            std::cout << "CSV file with ONNX inference times closed. Total inferences: " << inference_count_ << std::endl;
        }
    }

    // predict keypoints using cpu
    void pred_cpu(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts) {
        double inference_time_seconds = 0.0;
        
        // 完全對應LibTorch的圖像預處理
        if (eo.channels() != 1 || ir.channels() != 1) {
            throw std::runtime_error("ImageAlignONNXImpl::pred: eo and ir must be single channel images");
        }
        
        // 根據 pred_mode 決定輸入型別
        bool use_fp16 = (param_.pred_mode == "fp16");
        std::cout << "debug: Using " << (use_fp16 ? "FP16" : "FP32") << " input (pred_mode=" << param_.pred_mode << ")" << std::endl;
        
        cv::Mat eo_float, ir_float;
        eo.convertTo(eo_float, CV_32F, 1.0 / 255.0, 0.0);
        ir.convertTo(ir_float, CV_32F, 1.0 / 255.0, 0.0);

        size_t input_size = param_.pred_height * param_.pred_width;
        
        // 創建張量形狀 [1, 1, H, W]
        std::vector<int64_t> input_shape = {1, 1, param_.pred_height, param_.pred_width};
        
        // 創建輸入張量向量
        std::vector<Ort::Value> inputs;
        
        if (use_fp16) {
            // ===== FP16 模式：將輸入轉換為 FP16 =====
            std::vector<Ort::Float16_t> eo_fp16_data(input_size);
            std::vector<Ort::Float16_t> ir_fp16_data(input_size);

            // 將 FP32 轉換為 FP16
            const float* eo_ptr = eo_float.ptr<float>();
            const float* ir_ptr = ir_float.ptr<float>();
            
            for (size_t i = 0; i < input_size; i++) {
                eo_fp16_data[i] = Ort::Float16_t(eo_ptr[i]);
                ir_fp16_data[i] = Ort::Float16_t(ir_ptr[i]);
            }
            
            std::cout << "debug: Converted input data to FP16 format" << std::endl;

            // 創建 FP16 張量（注意：data 必須在 Run 之前有效，所以使用靜態變數）
            static std::vector<Ort::Float16_t> eo_fp16_static, ir_fp16_static;
            eo_fp16_static = std::move(eo_fp16_data);
            ir_fp16_static = std::move(ir_fp16_data);
            
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                *memory_info_, eo_fp16_static.data(), input_size, input_shape.data(), 4));
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                *memory_info_, ir_fp16_static.data(), input_size, input_shape.data(), 4));
        } else {
            
            std::cout << "777777777777777777777777777777777777" << std::endl;
            // ===== FP32 模式：使用 FP32 輸入 =====
            std::vector<float> eo_float32_data(input_size);
            std::vector<float> ir_float32_data(input_size);

            // 使用 .ptr<float>() 直接存取資料
            for (size_t i = 0; i < input_size; i++) {
                eo_float32_data[i] = eo_float.ptr<float>()[i];
                ir_float32_data[i] = ir_float.ptr<float>()[i];
            }
            
            // 創建 FP32 張量（同樣使用靜態變數）
            static std::vector<float> eo_fp32_static, ir_fp32_static;
            eo_fp32_static = std::move(eo_float32_data);
            ir_fp32_static = std::move(ir_float32_data);
            
            inputs.push_back(Ort::Value::CreateTensor<float>(
                *memory_info_, eo_fp32_static.data(), input_size, input_shape.data(), 4));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                *memory_info_, ir_fp32_static.data(), input_size, input_shape.data(), 4));
        }

        // 開始推理計時
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        // 創建運行選項
        Ort::RunOptions run_options;
        run_options.SetRunLogSeverityLevel(3);
        
        std::cout << "debug: Running ONNX model inference (pred_mode=" << param_.pred_mode << ")..." << std::endl;
        
        // 執行模型推理
        auto pred = session_->Run(run_options, input_names_.data(), inputs.data(), 2, 
                                output_names_.data(), output_names_.size());
        
        // 結束推理計時
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
        inference_time_seconds = inference_duration.count() / 1000000.0;
        
        std::cout << "ONNX Inference time (" << param_.pred_mode << "): " << inference_time_seconds << " seconds" << std::endl;

        
        // 新模型只返回 2 個輸出：mkpts0 和 mkpts1 (int32 類型)
        const int32_t *eo_res = pred[0].GetTensorMutableData<int32_t>();
        const int32_t *ir_res = pred[1].GetTensorMutableData<int32_t>();
        
        // 獲取輸出維度 [1200, 2]
        auto eo_shape = pred[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_points = static_cast<int>(eo_shape[0]);  // 應該是 1200
        
        eo_mkpts.clear();
        ir_mkpts.clear();

        // 遍歷所有點，過濾掉座標為 (0, 0) 的無效點
        for (int i = 0, pt = 0; i < num_points; i++, pt += 2) {
            int eo_x = static_cast<int>(eo_res[pt]);
            int eo_y = static_cast<int>(eo_res[pt + 1]);
            int ir_x = static_cast<int>(ir_res[pt]);
            int ir_y = static_cast<int>(ir_res[pt + 1]);
            
            // 跳過座標為 (0, 0) 的無效點
            if (eo_x == 0 && eo_y == 0) {
                continue;
            }
            
            eo_mkpts.push_back(cv::Point2i(eo_x, eo_y));
            ir_mkpts.push_back(cv::Point2i(ir_x, ir_y));
        }
        
        std::cout << "Extracted " << eo_mkpts.size() << " valid feature point pairs (pred_mode=" << param_.pred_mode << ")" << std::endl;
        
        // 記錄推理時間到CSV（對應LibTorch的writeTimingToCSV）
        inference_count_++;
        if (csv_file_.is_open()) {
            std::string image_name = current_image_name_.empty() ? 
                "----" : current_image_name_;
            if(image_name=="----"){
                return;
            }
            csv_file_ << image_name << "," << inference_time_seconds << "," 
                     << eo_mkpts.size() << std::endl;
            csv_file_.flush();
        }
    }

    // alias for pred_cpu
    void pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts) {
        pred_cpu(eo, ir, eo_mkpts, ir_mkpts);
    }

    // align with last H - 與LibTorch版本保持一致
    void align(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, cv::Mat &H) {
        // predict keypoints
        pred(eo, ir, eo_pts, ir_pts);
        
        // 返回單位矩陣，讓main.cpp處理homography計算（與LibTorch版本一致）
        H = cv::Mat::eye(3, 3, CV_64F);
        std::cout << "Feature point extraction complete. Found " << eo_pts.size() << " points." << std::endl;
    }

    bool align(const cv::Mat& eo, const cv::Mat& ir,
              std::vector<cv::Point2i>& eo_pts,
              std::vector<cv::Point2i>& ir_pts,
              cv::Mat& H) override {
        
        try {
            cv::Mat eo_copy = eo.clone();
            cv::Mat ir_copy = ir.clone();
            align(eo_copy, ir_copy, eo_pts, ir_pts, H);
            return !H.empty() && cv::determinant(H) > 1e-6;
        } catch (const std::exception& e) {
            std::cerr << "Error in alignment: " << e.what() << std::endl;
            return false;
        }
    }

    void set_current_image_name(const std::string& name) override {
        current_image_name_ = name;
    }
};

ImageAlignONNX::ptr ImageAlignONNX::create_instance(const Param& param) {
    return std::make_shared<ImageAlignONNXImpl>(param);
}

}

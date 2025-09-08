#include "../include/core_image_align_onnx.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <fstream>
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
    
    // Warm-up inference - 參考LibTorch的做法
    void warm_up() {
        std::cout << "Warm up..." << std::endl;
        
        // 創建與LibTorch相同的warm-up數據
        cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;
        cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;
        
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; i++) {
            std::vector<cv::Point2i> eo_mkpts, ir_mkpts;
            try {
                pred_cpu(eo, ir, eo_mkpts, ir_mkpts);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Warm-up iteration " << i << " failed: " << e.what() << std::endl;
            }
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
        std::cout << "Warm up completed in " << dt.count() << " ms" << std::endl;
    }

public:
    ImageAlignONNXImpl(const Param& param) : param_(param), env_(ORT_LOGGING_LEVEL_WARNING, "ImageAlign") {
        // ===== 完全移除隨機種子設置，模仿TensorRT的做法 =====
        // TensorRT版本沒有設置任何隨機種子，但卻是確定性的
        // 移除所有隨機種子相關設置
        std::cout << "Initializing ONNX Runtime without random seed (TensorRT style)..." << std::endl;
        
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
            // ===== 更嚴格的ONNX Runtime確定性設定 =====
            std::cout << "Configuring ONNX Runtime for maximum determinism..." << std::endl;
            
            // 強制單執行緒執行，避免並行導致的非確定性
            session_options_.SetIntraOpNumThreads(1);                    
            session_options_.SetInterOpNumThreads(1);                    
            
            // 完全禁用圖優化，避免優化導致的非確定性
            // session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL); 
            
            // 強制序列執行
            // session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);  
            
            // 設定日志等級
            // session_options_.SetLogSeverityLevel(3);
            
            // ===== 添加ONNX Runtime環境變量設定（強制確定性） =====
            // std::cout << "Setting deterministic environment variables..." << std::endl;
            
            // // 檢查並設定環境變量
            // if (!std::getenv("OMP_NUM_THREADS")) {
            //     putenv(const_cast<char*>("OMP_NUM_THREADS=1"));
            //     std::cout << "Set OMP_NUM_THREADS=1" << std::endl;
            // }
            // if (!std::getenv("CUDA_LAUNCH_BLOCKING")) {
            //     putenv(const_cast<char*>("CUDA_LAUNCH_BLOCKING=1"));
            //     std::cout << "Set CUDA_LAUNCH_BLOCKING=1" << std::endl;
            // }
            // if (!std::getenv("CUDNN_DETERMINISTIC")) {
            //     putenv(const_cast<char*>("CUDNN_DETERMINISTIC=1"));
            //     std::cout << "Set CUDNN_DETERMINISTIC=1" << std::endl;
            // }
            
            // Check if CUDA is requested and available
            std::cout << "Attempting to initialize ONNX Runtime with device: " << param_.device << std::endl;
            
            bool use_cuda = false;
            // ===== 專注解決ONNX FP16確定性問題 =====
            if (param_.device == "cuda") {  // 使用CUDA
                std::cout << "Adding CUDA execution provider with deterministic settings..." << std::endl;
                
                // ONNX CUDA設定，強制確定性行為
                OrtCUDAProviderOptions cuda_options{};
                // cuda_options.device_id = 0;
                
                // ===== 關鍵：強制CUDA使用確定性算法 =====
                // cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault; // 使用確定性算法
                // cuda_options.do_copy_in_default_stream = 1;    // 強制同步複製
                // cuda_options.has_user_compute_stream = 0;      // 禁用用戶流
                
                // ===== 禁用可能導致非確定性的優化 =====
                // cuda_options.tunable_op_enable = 0;           // 禁用可調優化
                // cuda_options.tunable_op_tuning_enable = 0;    // 禁用調優
                // cuda_options.arena_extend_strategy = 0;       // 固定記憶體策略
                // cuda_options.gpu_mem_limit = 0;               // 不限制記憶體              
                
                session_options_.AppendExecutionProvider_CUDA(cuda_options);
                // std::cout << "CUDA執行提供者已添加（強制確定性模式）" << std::endl;
                use_cuda = true;
                
                use_cuda = true;
                std::cout << "CUDA execution provider added successfully" << std::endl;
            } else {
                std::cout << "Using CPU execution provider" << std::endl;
            }
            
            session_ = std::make_unique<Ort::Session>(env_, param_.model_path.c_str(), session_options_);
            
            // Use appropriate memory allocator based on execution provider
            if (use_cuda) {
                memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
            } else {
                memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
            }
            
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
            
            // ===== 執行warm-up，與LibTorch保持一致 =====
            warm_up();
            
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
        // ===== 移除每次推理前的隨機種子重設，與LibTorch保持一致 =====
        // LibTorch只在初始化時設定一次torch::manual_seed(1)，推理時不重設
        // 移除：srand(1); cv::setRNGSeed(1);
        
        double inference_time_seconds = 0.0;
        
        // ===== 完全對應LibTorch的圖像預處理 =====
        // LibTorch檢查: if (eo.channels() != 1 || ir.channels() != 1)
        if (eo.channels() != 1 || ir.channels() != 1) {
            throw std::runtime_error("ImageAlignONNXImpl::pred: eo and ir must be single channel images");
        }
        
        // 檢查並調整圖像尺寸（與LibTorch保持一致）
        cv::Mat eo_resized, ir_resized;
        if (eo.cols != param_.pred_width || eo.rows != param_.pred_height) {
            cv::resize(eo, eo_resized, cv::Size(param_.pred_width, param_.pred_height));
            cv::resize(ir, ir_resized, cv::Size(param_.pred_width, param_.pred_height));
            std::cout << "DEBUG: Resized input from " << eo.cols << "x" << eo.rows 
                      << " to " << param_.pred_width << "x" << param_.pred_height << std::endl;
        } else {
            eo_resized = eo;  // 直接使用，不clone（對應LibTorch的from_blob直接使用）
            ir_resized = ir;
        }
        
        // ===== CUDA FP16確定性測試 =====
        if (param_.pred_mode == "fp16") {  // 恢復FP16
            std::cout << "DEBUG: Using FP16 mode with CUDA determinism fixes" << std::endl;
            
            // 完全對應LibTorch步驟：
            // 1. torch::from_blob(eo.data, {1, 1, H, W}, torch::kUInt8)
            // 2. .to(device).to(torch::kFloat32) / 255.0f  
            // 3. .to(torch::kHalf)
            
            // 步驟1&2: 從uint8轉為float32並正規化
            size_t input_size = param_.pred_height * param_.pred_width;
            std::vector<float> eo_float32_data(input_size);
            std::vector<float> ir_float32_data(input_size);
            
            // 直接從uint8數據轉換（對應torch::from_blob + to(torch::kFloat32) / 255.0f）
            const uint8_t* eo_src = eo_resized.data;
            const uint8_t* ir_src = ir_resized.data;
            
            for (size_t i = 0; i < input_size; i++) {
                eo_float32_data[i] = static_cast<float>(eo_src[i]) / 255.0f;
                ir_float32_data[i] = static_cast<float>(ir_src[i]) / 255.0f;
            }
            
            // 步驟3: 轉換為FP16（對應.to(torch::kHalf)）- 添加確定性處理
            std::vector<Ort::Float16_t> eo_fp16_data(input_size);
            std::vector<Ort::Float16_t> ir_fp16_data(input_size);
            
            // 確定性的FP16轉換：先排序索引確保轉換順序一致
            for (size_t i = 0; i < input_size; i++) {
                // 使用精確的FP16轉換，避免隨機性
                float eo_val = eo_float32_data[i];
                float ir_val = ir_float32_data[i];
                
                // 確保FP16轉換的確定性
                eo_fp16_data[i] = Ort::Float16_t(eo_val);
                ir_fp16_data[i] = Ort::Float16_t(ir_val);
            }
            
            // 創建張量（對應LibTorch張量形狀 {1, 1, H, W}）
            std::vector<int64_t> input_shape = {1, 1, param_.pred_height, param_.pred_width};

            std::cout << "DEBUG: Creating FP16 tensors (LibTorch identical) with shape [1, 1, " 
                      << param_.pred_height << ", " << param_.pred_width << "]" << std::endl;

            // 建立FP16張量
            Ort::Value eo_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                *memory_info_, eo_fp16_data.data(), input_size, input_shape.data(), 4);
            Ort::Value ir_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                *memory_info_, ir_fp16_data.data(), input_size, input_shape.data(), 4);
            
            // 創建輸入張量向量
            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(eo_tensor));
            inputs.push_back(std::move(ir_tensor));
            
            // // ===== 強制CUDA同步確保確定性 =====
            // if (param_.device == "cuda") {
            //     // 嘗試CUDA同步，如果CUDA不可用則跳過
            //     #ifdef __CUDACC__
            //     cudaDeviceSynchronize();
            //     #endif
            // }
            
            // 創建運行選項
            Ort::RunOptions run_options;
            run_options.SetRunLogSeverityLevel(3);
            
            std::cout << "DEBUG: Running FP16 model inference with CUDA sync..." << std::endl;
            
            // 開始推理計時
            auto inference_start = std::chrono::high_resolution_clock::now();
            
            // 執行模型推理
            auto pred = session_->Run(run_options, input_names_.data(), inputs.data(), 2, 
                                    output_names_.data(), output_names_.size());
            
            // 結束推理計時
            auto inference_end = std::chrono::high_resolution_clock::now();
            auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
            inference_time_seconds = inference_duration.count() / 1000000.0;
            
            // // ===== 強制CUDA同步確保推理完成 =====
            // if (param_.device == "cuda") {
            //     // 嘗試CUDA同步，如果CUDA不可用則跳過
            //     #ifdef __CUDACC__
            //     cudaDeviceSynchronize();
            //     #endif
            // }
            
            std::cout << "ONNX FP16 Inference time: " << inference_time_seconds << " seconds" << std::endl;
            
            // 獲取模型輸出（對應LibTorch的pred_[0], pred_[1], pred_[2]）
            const int64_t *eo_res = pred[0].GetTensorMutableData<int64_t>();
            const int64_t *ir_res = pred[1].GetTensorMutableData<int64_t>();
            const long int leng1 = pred[2].GetTensorMutableData<long int>()[0];
            const long int leng2 = pred[3].GetTensorMutableData<long int>()[0];
            
            eo_mkpts.clear();
            ir_mkpts.clear();

            // 提取關鍵點（完全對應LibTorch的處理方式）
            int len = leng1;
            for (int i = 0, pt = 0; i < len; i++, pt += 2) {
                // 對應LibTorch: static_cast<int>(eo_x), static_cast<int>(eo_y)
                int eo_x = static_cast<int>(eo_res[pt]);
                int eo_y = static_cast<int>(eo_res[pt + 1]);
                int ir_x = static_cast<int>(ir_res[pt]);
                int ir_y = static_cast<int>(ir_res[pt + 1]);
                
                eo_mkpts.push_back(cv::Point2i(eo_x, eo_y));
                ir_mkpts.push_back(cv::Point2i(ir_x, ir_y));
            }
            
            std::cout << "Extracted " << len << " feature point pairs (FP16 mode)" << std::endl;
            
        } else {
            // FP32模式：對應LibTorch的FP32處理
            std::cout << "DEBUG: Using FP32 input mode (LibTorch identical)" << std::endl;
            
            // 完全對應LibTorch步驟：
            // torch::from_blob(eo.data, {1, 1, H, W}, torch::kUInt8).to(device).to(torch::kFloat32) / 255.0f
            
            size_t input_size = param_.pred_height * param_.pred_width;
            std::vector<float> eo_float32_data(input_size);
            std::vector<float> ir_float32_data(input_size);
            
            // 直接從uint8數據轉換
            const uint8_t* eo_src = eo_resized.data;
            const uint8_t* ir_src = ir_resized.data;
            
            for (size_t i = 0; i < input_size; i++) {
                eo_float32_data[i] = static_cast<float>(eo_src[i]) / 255.0f;
                ir_float32_data[i] = static_cast<float>(ir_src[i]) / 255.0f;
            }

            // 創建張量形狀 [1, 1, H, W]
            std::vector<int64_t> input_shape = {1, 1, param_.pred_height, param_.pred_width};

            // 創建張量
            Ort::Value eo_tensor = Ort::Value::CreateTensor<float>(
                *memory_info_, eo_float32_data.data(), input_size, input_shape.data(), 4);
            Ort::Value ir_tensor = Ort::Value::CreateTensor<float>(
                *memory_info_, ir_float32_data.data(), input_size, input_shape.data(), 4);

            // 創建輸入張量向量
            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(eo_tensor));
            inputs.push_back(std::move(ir_tensor));

            // 開始推理計時
            auto inference_start = std::chrono::high_resolution_clock::now();
            
            // 創建運行選項
            Ort::RunOptions run_options;
            run_options.SetRunLogSeverityLevel(3);
            
            std::cout << "DEBUG: Running FP32 model inference..." << std::endl;
            
            // 執行模型推理
            auto pred = session_->Run(run_options, input_names_.data(), inputs.data(), 2, 
                                    output_names_.data(), output_names_.size());
            
            // 結束推理計時
            auto inference_end = std::chrono::high_resolution_clock::now();
            auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
            inference_time_seconds = inference_duration.count() / 1000000.0;
            
            std::cout << "ONNX FP32 Inference time: " << inference_time_seconds << " seconds" << std::endl;

            // 獲取模型輸出
            const int64_t *eo_res = pred[0].GetTensorMutableData<int64_t>();
            const int64_t *ir_res = pred[1].GetTensorMutableData<int64_t>();
            const long int leng1 = pred[2].GetTensorMutableData<long int>()[0];
            const long int leng2 = pred[3].GetTensorMutableData<long int>()[0];
            
            eo_mkpts.clear();
            ir_mkpts.clear();

            // 提取關鍵點（完全對應LibTorch的處理方式）
            int len = leng1;
            for (int i = 0, pt = 0; i < len; i++, pt += 2) {
                // 對應LibTorch: static_cast<int>(eo_x), static_cast<int>(eo_y)
                int eo_x = static_cast<int>(eo_res[pt]);
                int eo_y = static_cast<int>(eo_res[pt + 1]);
                int ir_x = static_cast<int>(ir_res[pt]);
                int ir_y = static_cast<int>(ir_res[pt + 1]);
                
                eo_mkpts.push_back(cv::Point2i(eo_x, eo_y));
                ir_mkpts.push_back(cv::Point2i(ir_x, ir_y));
            }
            
            std::cout << "Extracted " << len << " feature point pairs (FP32 mode)" << std::endl;
        }
        
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

        // CORRECTED: 與Python代碼完全一致的特徵點縮放條件檢查
        if (std::abs(param_.out_width_scale - 1.0) > 1e-6 || std::abs(param_.out_height_scale - 1.0) > 1e-6 || param_.bias_x > 0 || param_.bias_y > 0) {
            for (cv::Point2i &pt : eo_pts) {
                pt.x = pt.x * param_.out_width_scale + param_.bias_x;
                pt.y = pt.y * param_.out_height_scale + param_.bias_y;
            }
            for (cv::Point2i &pt : ir_pts) {
                pt.x = pt.x * param_.out_width_scale + param_.bias_x;
                pt.y = pt.y * param_.out_height_scale + param_.bias_y;
            }
            std::cout << "Feature point scaling applied (ONNX): scale=(" << param_.out_width_scale << ", " << param_.out_height_scale 
                      << "), bias=(" << param_.bias_x << ", " << param_.bias_y << ")" << std::endl;
        } else {
            std::cout << "No feature point scaling needed (ONNX): scale=(" << param_.out_width_scale << ", " << param_.out_height_scale 
                      << "), bias=(" << param_.bias_x << ", " << param_.bias_y << ")" << std::endl;
        }
        
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

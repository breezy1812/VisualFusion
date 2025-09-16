#include "../include/core_image_align_tensorrt.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <cuda_fp16.h>  // 添加 FP16 支援
// pyfp16 onnxfp16 trtfp16
// TensorRT Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "debug:" << msg << std::endl;
        }
    }
};

namespace core {

// PIMPL Idiom: Implementation class
class ImageAlignTensorRTImpl : public ImageAlignTensorRT {
private:
    Param param_;
    Logger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    
    // CSV logging for inference time
    std::ofstream csv_file_;
    std::string current_image_name_ = "";  // 當前圖片名稱

    // Helper to load engine from file
    bool loadEngine(const std::string& engine_path) {
        std::cout << "debug: Loading TensorRT engine from: " << engine_path << std::endl;
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "debug: ERROR: Could not open engine file: " << engine_path << std::endl;
            return false;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            std::cerr << "debug: ERROR: Could not read engine file." << std::endl;
            return false;
        }

        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_) {
            std::cerr << "debug: ERROR: Failed to create TensorRT runtime." << std::endl;
            return false;
        }

        engine_ = runtime_->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
        if (!engine_) {
            std::cerr << "debug: ERROR: Failed to deserialize CUDA engine." << std::endl;
            return false;
        }
        
        std::cout << "debug: TensorRT engine loaded successfully." << std::endl;
        return true;
    }

public:
    // Constructor
    ImageAlignTensorRTImpl(const Param& param) : ImageAlignTensorRT(param), param_(param) {
        std::cout << "debug: Initializing ImageAlignTensorRT with pred_mode=" << param_.pred_mode << "..." << std::endl;
        
        // 初始化 CSV 文件
        csv_file_.open("tensorrt_inference_times.csv", std::ios::app);
        if (csv_file_.is_open()) {
            // 檢查文件是否為空，如果是則添加標頭
            csv_file_.seekp(0, std::ios::end);
            if (csv_file_.tellp() == 0) {
                csv_file_ << "Image_Name,Inference_Time_Seconds,Features_Count" << std::endl;
            }
        }
        
        if (!loadEngine(param_.engine_path)) {
            throw std::runtime_error("Failed to load TensorRT engine.");
        }

        context_ = engine_->createExecutionContext();
        if (!context_) {
            throw std::runtime_error("Failed to create TensorRT execution context.");
        }

        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream.");
        }
        
        std::cout << "debug: ImageAlignTensorRT initialized successfully with pred_mode=" << param_.pred_mode << std::endl;
        
        // 執行 warmup
        warm_up();
    }

    // Destructor
    ~ImageAlignTensorRTImpl() {
        std::cout << "debug: Releasing ImageAlignTensorRT resources..." << std::endl;
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
        if (stream_) cudaStreamDestroy(stream_);
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
        if (runtime_) runtime_->destroy();
        std::cout << "debug: Resources released." << std::endl;
    }

    // Warmup function - 與LibTorch版本一致
    void warm_up() {
        std::cout<<"******************************"<<std::endl;
        std::cout << "debug: TensorRT Warm up..." << std::endl;
        std::cout<<"******************************"<<std::endl;

        cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;
        cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;

        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; i++) {
            std::vector<cv::Point2i> eo_mkpts, ir_mkpts;
            pred(eo, ir, eo_mkpts, ir_mkpts);
        }

        const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
        const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

        std::cout << "debug: TensorRT Warm up done in " << std::fixed << std::setprecision(2) << period << " s" << std::endl;
    }

    // 更新圖片名稱的方法
    void set_current_image_name(const std::string& image_name) {
        current_image_name_ = image_name;
    }

    // Main alignment function
    void align(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_pts, std::vector<cv::Point2i>& ir_pts, cv::Mat& H) override {
        std::cout << "debug: Starting alignment process for image: " << current_image_name_ << std::endl;
        pred(eo, ir, eo_pts, ir_pts);

        if (eo_pts.size() < 4) {
            std::cout << "debug: Not enough points to compute homography. Found only " << eo_pts.size() << " points." << std::endl;
            H = cv::Mat::eye(3, 3, CV_64F);
            return;
        }

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
            std::cout << "debug: Feature point scaling applied (TensorRT): scale=(" << param_.out_width_scale << ", " << param_.out_height_scale 
                      << "), bias=(" << param_.bias_x << ", " << param_.bias_y << ")" << std::endl;
        } else {
            std::cout << "debug: No feature point scaling needed (TensorRT): scale=(" << param_.out_width_scale << ", " << param_.out_height_scale 
                      << "), bias=(" << param_.bias_x << ", " << param_.bias_y << ")" << std::endl;
        }

        // 與LibTorch版本保持一致，不在這裡做RANSAC，只返回特徵點
        // RANSAC處理將在main.cpp中進行
        H = cv::Mat::eye(3, 3, CV_64F);  // 返回單位矩陣，讓main.cpp處理homography計算
        std::cout << "debug: Feature point extraction complete. Found " << eo_pts.size() << " points." << std::endl;
    }

    // Pre-processing and inference prediction
    void pred(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_mkpts, std::vector<cv::Point2i>& ir_mkpts) {
        std::cout << "debug: Starting prediction..." << std::endl;
        
        // 1. Pre-processing - 使用默認插值方法
        cv::Mat eo_resized, ir_resized;
        cv::resize(eo, eo_resized, cv::Size(param_.pred_width, param_.pred_height));
        cv::resize(ir, ir_resized, cv::Size(param_.pred_width, param_.pred_height));

        cv::Mat eo_gray, ir_gray;
        
        // 檢查圖像是否已經是灰度圖
        if (eo_resized.channels() == 3) {
            cv::cvtColor(eo_resized, eo_gray, cv::COLOR_BGR2GRAY);
        } else {
            eo_gray = eo_resized.clone();
        }
        
        if (ir_resized.channels() == 3) {
            cv::cvtColor(ir_resized, ir_gray, cv::COLOR_BGR2GRAY);  
        } else {
            ir_gray = ir_resized.clone();
        }

        // CORRECTED: 完全按照Python代碼邏輯進行處理
        // Python: img0_tensor = torch.from_numpy(eo_pred)[None][None].to(device, dtype=fpMode) / 255.
        
        // 1. 確保圖像是uint8格式（對應Python numpy array）
        cv::Mat eo_uint8, ir_uint8;
        eo_gray.convertTo(eo_uint8, CV_8U);
        ir_gray.convertTo(ir_uint8, CV_8U);
        
        cv::Mat eo_float, ir_float;
        // 修改：統一使用FP32格式輸入數據
        // 轉換流程：PyTorch(FP32) → ONNX(FP32) → TensorRT(FP16)
        // TensorRT引擎內部使用FP16，但輸入數據保持FP32格式
        std::cout << "debug: Converting images to FP32 format (unified input)..." << std::endl;
        eo_uint8.convertTo(eo_float, CV_32F, 1.0f / 255.0f);
        ir_uint8.convertTo(ir_float, CV_32F, 1.0f / 255.0f);
        
        if (param_.pred_mode == "fp16") {
            std::cout << "debug: Using FP16 TensorRT engine with FP32 input data..." << std::endl;
        } else {
            std::cout << "debug: Using FP32 TensorRT engine with FP32 input data..." << std::endl;
        }

        // 統一處理：無論TensorRT引擎是FP16還是FP32，都使用FP32輸入數據
        int valid_points_count = 0;
        std::vector<float> eo_data(param_.pred_width * param_.pred_height);
        std::vector<float> ir_data(param_.pred_width * param_.pred_height);
        
        // 將FP32圖像數據拷貝到向量中
        memcpy(eo_data.data(), eo_float.data, eo_data.size() * sizeof(float));
        memcpy(ir_data.data(), ir_float.data, ir_data.size() * sizeof(float));
        
        std::cout << "debug: Pre-processing complete. Using FP32 input data. Image size: " 
                  << param_.pred_width << "x" << param_.pred_height << std::endl;

        // 執行推理 - 統一使用FP32數據接口
        std::cout << "debug: [pred] Before runInference. eo_mkpts size: " << eo_mkpts.size() 
                  << ", ir_mkpts size: " << ir_mkpts.size() << std::endl;
        
        auto inference_start = std::chrono::high_resolution_clock::now();
        bool success = runInference(eo_data, ir_data, eo_mkpts, ir_mkpts, valid_points_count);
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
        double inference_time_seconds = inference_duration.count() / 1000000.0;
        
        std::cout << "debug: [pred] After runInference. eo_mkpts size: " << eo_mkpts.size() 
                  << ", ir_mkpts size: " << ir_mkpts.size() << ", valid_points_count: " << valid_points_count << std::endl;
        std::cout << "Inference time: " << inference_time_seconds << " seconds" << std::endl;
        
        // 記錄到CSV
        if (csv_file_.is_open()) {
            std::string image_name = current_image_name_.empty() ? "----" : current_image_name_;
            if(image_name != "----") {
                csv_file_ << image_name << "," << std::fixed << std::setprecision(6) 
                         << inference_time_seconds << "," << eo_mkpts.size() << std::endl;
                csv_file_.flush();
            }
        }

        if (!success) {
            std::cerr << "debug: ERROR: Inference failed." << std::endl;
            eo_mkpts.clear();
            ir_mkpts.clear();
            return;
        }
        
        std::cout << "debug: Inference successful. Total valid points processed." << std::endl;

        // 3. Post-processing: 使用前leng個特徵點，後面的都是(0,0)不採用
        if (valid_points_count > 0 && valid_points_count <= (int)eo_mkpts.size()) {
            // 只保留前leng個有效特徵點
            eo_mkpts.resize(valid_points_count);
            ir_mkpts.resize(valid_points_count);
            std::cout << "debug: [pred] Resized to " << eo_mkpts.size() << " keypoints based on leng=" << valid_points_count << std::endl;
        } else if (valid_points_count <= 0) {
            std::cout << "debug: WARNING: leng=" << valid_points_count << " is invalid. Clearing results." << std::endl;
            eo_mkpts.clear();
            ir_mkpts.clear();
        } else {
            std::cout << "debug: WARNING: leng=" << valid_points_count << " is greater than detected points " << eo_mkpts.size() << ". Using all detected points." << std::endl;
        }
        
        std::cout << "debug: Post-processing complete. Final keypoint count: " << eo_mkpts.size() << std::endl;
    }

    // The core inference logic
    bool runInference(const std::vector<float>& eo_data, const std::vector<float>& ir_data, 
                      std::vector<cv::Point2i>& eo_kps, std::vector<cv::Point2i>& ir_kps, int& leng1) {
        
        const int num_bindings = engine_->getNbBindings();
        if (num_bindings != 6) { // 2 inputs + 4 outputs
            std::cerr << "debug: ERROR: Expected 6 bindings, but got " << num_bindings << std::endl;
            return false;
        }

        std::vector<void*> buffers(num_bindings);
        
        // Get binding indices and allocate GPU buffers
        for (int i = 0; i < num_bindings; ++i) {
            auto dims = engine_->getBindingDimensions(i);
            const char* binding_name = engine_->getBindingName(i);
            nvinfer1::DataType dtype = engine_->getBindingDataType(i);
            size_t element_size = 0;
            std::string type_str = "UNKNOWN";

            // Determine element size based on data type
            switch (dtype) {
                case nvinfer1::DataType::kFLOAT: 
                    element_size = sizeof(float); 
                    type_str = "FLOAT";
                    break;
                case nvinfer1::DataType::kINT32: 
                    element_size = sizeof(int32_t); 
                    type_str = "INT32";
                    break;
                // TensorRT 8.6 does not seem to have kINT64, so removing it.
                // case nvinfer1::DataType::kINT64: 
                //     element_size = sizeof(int64_t); 
                //     type_str = "INT64";
                //     break;
                default: 
                    std::cerr << "debug: ERROR: Unsupported data type for binding " << binding_name << " (Type: " << static_cast<int>(dtype) << ")" << std::endl;
                    // Free already allocated buffers before returning
                    for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                    return false;
            }
            std::cout << "debug: [runInference] Binding: " << i << ", Name: " << binding_name << ", Type: " << type_str << std::endl;
            
            size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>()) * element_size;
            
            if (cudaMalloc(&buffers[i], size) != cudaSuccess) {
                std::cerr << "debug: ERROR: CUDA memory allocation failed for binding " << i << " (" << binding_name << ")" << std::endl;
                // Free already allocated buffers before returning
                for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                return false;
            }
        }

        // Find specific binding indices by name
        int eo_img_idx = engine_->getBindingIndex("vi_img");
        int ir_img_idx = engine_->getBindingIndex("ir_img");
        int mkpt0_idx = engine_->getBindingIndex("mkpt0");
        int mkpt1_idx = engine_->getBindingIndex("mkpt1");
        int leng1_idx = engine_->getBindingIndex("leng1");
        int leng2_idx = engine_->getBindingIndex("leng2");

        if (eo_img_idx < 0 || ir_img_idx < 0 || mkpt0_idx < 0 || mkpt1_idx < 0 || leng1_idx < 0 || leng2_idx < 0) {
            std::cerr << "debug: ERROR: Could not find one or more required bindings by name." << std::endl;
            // Free all buffers before returning
            for(void* buf : buffers) cudaFree(buf);
            return false;
        }

        // Copy input data from host to device (GPU)
        cudaMemcpyAsync(buffers[eo_img_idx], eo_data.data(), eo_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(buffers[ir_img_idx], ir_data.data(), ir_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);

        // Execute the model
        std::cout << "debug: Executing model..." << std::endl;
        if (!context_->enqueueV2(buffers.data(), stream_, nullptr)) {
            std::cerr << "debug: ERROR: Failed to enqueue inference." << std::endl;
            return false;
        }

        // Host-side buffers for outputs
        auto mkpt0_dims = engine_->getBindingDimensions(mkpt0_idx);
        std::string dims_str = "debug: [runInference] mkpt0 dimensions (nbDims=" + std::to_string(mkpt0_dims.nbDims) + "): (";
        for (int j = 0; j < mkpt0_dims.nbDims; ++j) {
            dims_str += std::to_string(mkpt0_dims.d[j]) + (j < mkpt0_dims.nbDims - 1 ? ", " : "");
        }
        dims_str += ")";
        std::cout << dims_str << std::endl;

        size_t mkpt0_count = std::accumulate(mkpt0_dims.d, mkpt0_dims.d + mkpt0_dims.nbDims, 1, std::multiplies<size_t>());
        std::vector<int32_t> eo_kps_raw(mkpt0_count);

        auto mkpt1_dims = engine_->getBindingDimensions(mkpt1_idx);
        dims_str = "debug: [runInference] mkpt1 dimensions (nbDims=" + std::to_string(mkpt1_dims.nbDims) + "): (";
        for (int j = 0; j < mkpt1_dims.nbDims; ++j) {
            dims_str += std::to_string(mkpt1_dims.d[j]) + (j < mkpt1_dims.nbDims - 1 ? ", " : "");
        }
        dims_str += ")";
        std::cout << dims_str << std::endl;

        size_t mkpt1_count = std::accumulate(mkpt1_dims.d, mkpt1_dims.d + mkpt1_dims.nbDims, 1, std::multiplies<size_t>());
        std::vector<int32_t> ir_kps_raw(mkpt1_count);
        
        int32_t leng1_raw, leng2_raw;

        // Copy output data from device to host
        cudaMemcpyAsync(eo_kps_raw.data(), buffers[mkpt0_idx], eo_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(ir_kps_raw.data(), buffers[mkpt1_idx], ir_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(&leng1_raw, buffers[leng1_idx], sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(&leng2_raw, buffers[leng2_idx], sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);

        // Wait for all CUDA operations to complete
        cudaStreamSynchronize(stream_);
        std::cout << "debug: Model execution and data copy complete." << std::endl;

        // Process the raw output
        leng1 = leng1_raw;
        
        // The shape should be [1, 1200, 2]. Let's handle different nbDims cases.
        int num_keypoints = 0;
        int num_coords = 2; // Always 2 for (x, y)

        if (mkpt0_dims.nbDims == 3) { // Expected case: [1, 1200, 2]
            num_keypoints = mkpt0_dims.d[1];
            num_coords = mkpt0_dims.d[2];
        } else if (mkpt0_dims.nbDims == 2) { // Fallback for [1200, 2]
            num_keypoints = mkpt0_dims.d[0];
            num_coords = mkpt0_dims.d[1];
        } else {
            std::cerr << "debug: ERROR: Unexpected number of dimensions for keypoints: " << mkpt0_dims.nbDims << std::endl;
            return false;
        }
        
        if (num_coords != 2) {
             std::cerr << "debug: ERROR: Expected 2 coordinates per keypoint, but got " << num_coords << std::endl;
             return false;
        }

        eo_kps.clear();
        ir_kps.clear();
        // eo_kps.reserve(leng1);
        // ir_kps.reserve(leng1);
        std::cout << "debug: [runInference] Raw leng1=" << leng1  << std::endl;
        std::cout << "debug: [runInference] Parsing up to " << leng1 << " keypoints..." << std::endl;
        for (int i = 0; i < leng1; ++i) {
            // 使用round而非直接轉換，提高精度（與LibTorch版本一致）
            int x_eo = static_cast<int>(std::round(eo_kps_raw[i * num_coords + 0]));
            int y_eo = static_cast<int>(std::round(eo_kps_raw[i * num_coords + 1]));
            eo_kps.emplace_back(x_eo, y_eo);

            int x_ir = static_cast<int>(std::round(ir_kps_raw[i * num_coords + 0]));
            int y_ir = static_cast<int>(std::round(ir_kps_raw[i * num_coords + 1]));
            ir_kps.emplace_back(x_ir, y_ir);
        }
        
        std::cout << "debug: Raw leng1=" << leng1_raw << ", leng2=" << leng2_raw << std::endl;
        std::cout << "debug: [runInference] After parsing loop. Parsed " << eo_kps.size() << " raw keypoints." << std::endl;

        // Print first 5 keypoints to check their values
        for (int i = 0; i < std::min(5, (int)eo_kps.size()); ++i) {
            std::cout << "debug: [runInference] Raw KP " << i << ": EO(" << eo_kps[i].x << "," << eo_kps[i].y 
                      << "), IR(" << ir_kps[i].x << "," << ir_kps[i].y << ")" << std::endl;
        }

        // Free GPU buffers
        for (void* buf : buffers) {
            cudaFree(buf);
        }

        return true;
    }

    // FP16 inference function
    bool runInferenceFP16(const std::vector<__half>& eo_data, const std::vector<__half>& ir_data, 
                         std::vector<cv::Point2i>& eo_kps, std::vector<cv::Point2i>& ir_kps, int& leng1) {
        
        const int num_bindings = engine_->getNbBindings();
        if (num_bindings != 6) { // 2 inputs + 4 outputs
            std::cerr << "debug: ERROR: Expected 6 bindings for FP16, but got " << num_bindings << std::endl;
            return false;
        }

        std::vector<void*> buffers(num_bindings);
        
        // Get binding indices and allocate GPU buffers
        for (int i = 0; i < num_bindings; ++i) {
            auto dims = engine_->getBindingDimensions(i);
            const char* binding_name = engine_->getBindingName(i);
            nvinfer1::DataType dtype = engine_->getBindingDataType(i);
            size_t element_size = 0;
            std::string type_str = "UNKNOWN";

            // Determine element size based on data type
            switch (dtype) {
                case nvinfer1::DataType::kFLOAT: 
                    element_size = sizeof(float); 
                    type_str = "FLOAT";
                    break;
                case nvinfer1::DataType::kHALF:
                    element_size = sizeof(__half);
                    type_str = "HALF";
                    break;
                case nvinfer1::DataType::kINT32: 
                    element_size = sizeof(int32_t); 
                    type_str = "INT32";
                    break;
                default: 
                    std::cerr << "debug: ERROR: Unsupported data type for FP16 binding " << binding_name << " (Type: " << static_cast<int>(dtype) << ")" << std::endl;
                    // Free already allocated buffers before returning
                    for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                    return false;
            }
            std::cout << "debug: [runInferenceFP16] Binding: " << i << ", Name: " << binding_name << ", Type: " << type_str << std::endl;
            
            size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>()) * element_size;
            
            if (cudaMalloc(&buffers[i], size) != cudaSuccess) {
                std::cerr << "debug: ERROR: CUDA memory allocation failed for FP16 binding " << i << " (" << binding_name << ")" << std::endl;
                // Free already allocated buffers before returning
                for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                return false;
            }
        }

        // Find specific binding indices by name
        int eo_img_idx = engine_->getBindingIndex("vi_img");
        int ir_img_idx = engine_->getBindingIndex("ir_img");
        int mkpt0_idx = engine_->getBindingIndex("mkpt0");
        int mkpt1_idx = engine_->getBindingIndex("mkpt1");
        int leng1_idx = engine_->getBindingIndex("leng1");
        int leng2_idx = engine_->getBindingIndex("leng2");

        if (eo_img_idx < 0 || ir_img_idx < 0 || mkpt0_idx < 0 || mkpt1_idx < 0 || leng1_idx < 0 || leng2_idx < 0) {
            std::cerr << "debug: ERROR: Could not find one or more required bindings by name for FP16." << std::endl;
            // Free all buffers before returning
            for(void* buf : buffers) cudaFree(buf);
            return false;
        }

        // Copy FP16 input data from host to device (GPU)
        cudaMemcpyAsync(buffers[eo_img_idx], eo_data.data(), eo_data.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(buffers[ir_img_idx], ir_data.data(), ir_data.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);

        // Execute the model
        std::cout << "debug: Executing FP16 model..." << std::endl;
        if (!context_->enqueueV2(buffers.data(), stream_, nullptr)) {
            std::cerr << "debug: ERROR: Failed to enqueue FP16 inference." << std::endl;
            return false;
        }

        // Host-side buffers for outputs (same as FP32 version)
        auto mkpt0_dims = engine_->getBindingDimensions(mkpt0_idx);
        size_t mkpt0_count = std::accumulate(mkpt0_dims.d, mkpt0_dims.d + mkpt0_dims.nbDims, 1, std::multiplies<size_t>());
        std::vector<int32_t> eo_kps_raw(mkpt0_count);

        auto mkpt1_dims = engine_->getBindingDimensions(mkpt1_idx);
        size_t mkpt1_count = std::accumulate(mkpt1_dims.d, mkpt1_dims.d + mkpt1_dims.nbDims, 1, std::multiplies<size_t>());
        std::vector<int32_t> ir_kps_raw(mkpt1_count);
        
        int32_t leng1_raw, leng2_raw;

        // Copy output data from device to host
        cudaMemcpyAsync(eo_kps_raw.data(), buffers[mkpt0_idx], eo_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(ir_kps_raw.data(), buffers[mkpt1_idx], ir_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(&leng1_raw, buffers[leng1_idx], sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(&leng2_raw, buffers[leng2_idx], sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);

        // Wait for all CUDA operations to complete
        cudaStreamSynchronize(stream_);
        std::cout << "debug: FP16 model execution and data copy complete." << std::endl;

        // Process the raw output (same logic as FP32 version)
        leng1 = leng1_raw;
        
        // The shape should be [1, 1200, 2]. Let's handle different nbDims cases.
        int num_keypoints = 0;
        int num_coords = 2; // Always 2 for (x, y)

        if (mkpt0_dims.nbDims == 3) { // Expected case: [1, 1200, 2]
            num_keypoints = mkpt0_dims.d[1];
            num_coords = mkpt0_dims.d[2];
        } else if (mkpt0_dims.nbDims == 2) { // Fallback for [1200, 2]
            num_keypoints = mkpt0_dims.d[0];
            num_coords = mkpt0_dims.d[1];
        } else {
            std::cerr << "debug: ERROR: Unexpected number of dimensions for FP16 keypoints: " << mkpt0_dims.nbDims << std::endl;
            return false;
        }
        
        if (num_coords != 2) {
             std::cerr << "debug: ERROR: Expected 2 coordinates per FP16 keypoint, but got " << num_coords << std::endl;
             return false;
        }

        eo_kps.clear();
        ir_kps.clear();
        std::cout << "debug: [runInferenceFP16] Raw leng1=" << leng1 << std::endl;
        std::cout << "debug: [runInferenceFP16] Parsing up to " << leng1 << " keypoints..." << std::endl;
        for (int i = 0; i < leng1; ++i) {
            // 使用round而非直接轉換，提高精度（與LibTorch版本一致）
            int x_eo = static_cast<int>(std::round(eo_kps_raw[i * num_coords + 0]));
            int y_eo = static_cast<int>(std::round(eo_kps_raw[i * num_coords + 1]));
            eo_kps.emplace_back(x_eo, y_eo);

            int x_ir = static_cast<int>(std::round(ir_kps_raw[i * num_coords + 0]));
            int y_ir = static_cast<int>(std::round(ir_kps_raw[i * num_coords + 1]));
            ir_kps.emplace_back(x_ir, y_ir);
        }
        
        std::cout << "debug: FP16 Raw leng1=" << leng1_raw << ", leng2=" << leng2_raw << std::endl;
        std::cout << "debug: [runInferenceFP16] After parsing loop. Parsed " << eo_kps.size() << " raw keypoints." << std::endl;

        // Print first 5 keypoints to check their values
        for (int i = 0; i < std::min(5, (int)eo_kps.size()); ++i) {
            std::cout << "debug: [runInferenceFP16] Raw KP " << i << ": EO(" << eo_kps[i].x << "," << eo_kps[i].y 
                      << "), IR(" << ir_kps[i].x << "," << ir_kps[i].y << ")" << std::endl;
        }

        // Free GPU buffers
        for (void* buf : buffers) {
            cudaFree(buf);
        }

        return true;
    }
};

// Factory function to create an instance of the implementation
std::shared_ptr<ImageAlignTensorRT> ImageAlignTensorRT::create_instance(const Param& param) {
    return std::make_shared<ImageAlignTensorRTImpl>(param);
}

// Dummy implementations for the base class to allow linkage
ImageAlignTensorRT::ImageAlignTensorRT(const Param& param) {}
ImageAlignTensorRT::~ImageAlignTensorRT() {}

} // namespace core

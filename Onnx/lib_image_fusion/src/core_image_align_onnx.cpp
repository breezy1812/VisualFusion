#include "../include/core_image_align_onnx.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <fstream>
#include <onnxruntime_cxx_api.h>

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
    
    // Warm-up inference to optimize CUDA performance
    void warmup_inference() {
        try {
            // Create dummy input data with correct dimensions
            std::vector<int64_t> input_shape = {1, 1, param_.pred_height, param_.pred_width};
            size_t input_size = param_.pred_height * param_.pred_width;
            
            // Create dummy float data
            std::vector<float> dummy_data(input_size, 0.5f);
            
            // Create tensors
            Ort::Value eo_tensor = Ort::Value::CreateTensor<float>(
                *memory_info_, dummy_data.data(), input_size, input_shape.data(), 4);
            Ort::Value ir_tensor = Ort::Value::CreateTensor<float>(
                *memory_info_, dummy_data.data(), input_size, input_shape.data(), 4);
            
            // Create input tensor vector
            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(eo_tensor));
            inputs.push_back(std::move(ir_tensor));
            
            // Run warm-up inference
            auto pred = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), 
                                    inputs.data(), 2, output_names_.data(), output_names_.size());
                                    
            // Force synchronization for CUDA
            if (param_.device == "cuda") {
                // CUDA synchronization would happen here if needed
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Warm-up inference failed: " << e.what() << std::endl;
        }
    }

public:
    ImageAlignONNXImpl(const Param& param) : param_(param), env_(ORT_LOGGING_LEVEL_WARNING, "ImageAlign") {
        // Initialize CSV file for logging inference times
        csv_file_.open("onnx_inference_times.csv", std::ios::app);
        if (!csv_file_.is_open()) {
            std::cerr << "Warning: Could not open CSV file for writing inference times" << std::endl;
        } else {
            // Write header if file is empty/new
            csv_file_.seekp(0, std::ios::end);
            if (csv_file_.tellp() == 0) {
                csv_file_ << "Frame,Inference_Time_Seconds,Features_Count" << std::endl;
            }
        }
        
        // 檢查模型文件是否存在
        if (!std::experimental::filesystem::exists(param_.model_path)) {
            std::cerr << "FATAL ERROR: Model file not found: " << param_.model_path << std::endl;
            throw std::runtime_error("ONNX model file not found: " + param_.model_path);
        }
        
        try {
            // 優化 ONNX Runtime session 設定
            session_options_.SetIntraOpNumThreads(1);  // 減少執行緒數量以避免競爭
            session_options_.SetInterOpNumThreads(1);  // 單執行緒間操作
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC); // 使用基本優化避免過多 memcpy
            session_options_.EnableCpuMemArena();      // 啟用 CPU 記憶體池
            session_options_.EnableMemPattern();       // 啟用記憶體模式優化
            session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);  // 序列執行模式
            
            // 設定日志等級
            session_options_.SetLogSeverityLevel(3);   // 只顯示 ERROR，減少警告訊息
            
            // Check if CUDA is requested and available
            std::cout << "Attempting to initialize ONNX Runtime with device: " << param_.device << std::endl;
            
            bool use_cuda = false;
            if (param_.device == "cuda") {
                try {
                    // CUDA provider options - 優化設定以減少 memcpy
                    OrtCUDAProviderOptions cuda_options{};
                    cuda_options.device_id = 0;                    // Use first GPU
                    cuda_options.arena_extend_strategy = 0;        // 不擴展記憶體池
                    cuda_options.gpu_mem_limit = 0;                // 無限制，使用所有可用記憶體
                    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic; // 使用啟發式演算法，速度快
                    cuda_options.do_copy_in_default_stream = 1;    // 在預設串流中複製，減少同步開銷
                    cuda_options.has_user_compute_stream = 0;      // 不使用用戶串流
                    cuda_options.default_memory_arena_cfg = nullptr;
                    
                    session_options_.AppendExecutionProvider_CUDA(cuda_options);
                    
                    // 針對混合 CPU-GPU 執行進行優化
                    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
                    
                    use_cuda = true;
                    std::cout << "CUDA execution provider successfully added (optimized for mixed CPU-GPU)" << std::endl;
                } catch (const std::exception& cuda_e) {
                    std::cerr << "Warning: CUDA execution provider failed: " << cuda_e.what() << std::endl;
                    std::cerr << "Falling back to CPU execution provider" << std::endl;
                    use_cuda = false;
                }
            } else {
                std::cout << "Using CPU execution provider as requested" << std::endl;
            }
            
            session_ = std::make_unique<Ort::Session>(env_, param_.model_path.c_str(), session_options_);
            
            // Use appropriate memory allocator based on execution provider
            if (use_cuda) {
                // For CUDA, we still use CPU memory info for input/output tensors
                // The CUDA provider will handle GPU memory internally
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
        double inference_time_seconds = 0.0;  // Initialize inference time variable
        
        // resize input image to pred_width x pred_height
        cv::Mat eo_temp, ir_temp;
        cv::resize(eo, eo_temp, cv::Size(param_.pred_width, param_.pred_height));
        cv::resize(ir, ir_temp, cv::Size(param_.pred_width, param_.pred_height));

        // Convert to grayscale if necessary
        if (eo_temp.channels() == 3) {
            cv::cvtColor(eo_temp, eo_temp, cv::COLOR_BGR2GRAY);
        }
        if (ir_temp.channels() == 3) {
            cv::cvtColor(ir_temp, ir_temp, cv::COLOR_BGR2GRAY);
        }

        // normalize eo and ir to 0-1, and convert from cv::Mat to float
        eo_temp.convertTo(eo_temp, CV_32F, 1.0f / 255.0f);
        ir_temp.convertTo(ir_temp, CV_32F, 1.0f / 255.0f);

        // change the address type from uchar* to float*
        float *eo_data = reinterpret_cast<float *>(eo_temp.data);
        float *ir_data = reinterpret_cast<float *>(ir_temp.data);

        // create tensor shape [1, 1, H, W]
        std::vector<int64_t> input_shape = {1, 1, param_.pred_height, param_.pred_width};
        size_t input_size = param_.pred_height * param_.pred_width;

        // create tensor
        Ort::Value eo_tensor = Ort::Value::CreateTensor<float>(*memory_info_, eo_data, input_size, input_shape.data(), 4);
        Ort::Value ir_tensor = Ort::Value::CreateTensor<float>(*memory_info_, ir_data, input_size, input_shape.data(), 4);

        // create input tensor
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(eo_tensor));
        inputs.push_back(std::move(ir_tensor));

        // Start inference timing
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        // Create run options for better performance
        Ort::RunOptions run_options;
        run_options.SetRunLogSeverityLevel(3);  // Reduce logging overhead
        
        // run the model
        auto pred = session_->Run(run_options, input_names_.data(), inputs.data(), 2, output_names_.data(), output_names_.size());
        
        // End inference timing and calculate duration
        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
        inference_time_seconds = inference_duration.count() / 1000000.0;  // Convert to seconds
        
        std::cout << "ONNX Inference time: " << inference_time_seconds << " seconds" << std::endl;

        // get mkpts from the model output
        const int64_t *eo_res = pred[0].GetTensorMutableData<int64_t>();
        const int64_t *ir_res = pred[1].GetTensorMutableData<int64_t>();

        const long int leng1 = pred[2].GetTensorMutableData<long int>()[0];
        const long int leng2 = pred[3].GetTensorMutableData<long int>()[0];
        
        eo_mkpts.clear();
        ir_mkpts.clear();

        // push keypoints to eo_mkpts and ir_mkpts
        // int len = pred[0].GetTensorTypeAndShapeInfo().GetShape()[0];
        int len = leng1;
        for (int i = 0, pt = 0; i < len; i++, pt += 2) {
            // Scale points from model resolution to original image resolution
            float eo_x = eo_res[pt] * param_.out_width_scale + param_.bias_x;
            float eo_y = eo_res[pt + 1] * param_.out_height_scale + param_.bias_y;
            float ir_x = ir_res[pt] * param_.out_width_scale + param_.bias_x;
            float ir_y = ir_res[pt + 1] * param_.out_height_scale + param_.bias_y;
            
            eo_mkpts.push_back(cv::Point2i((int)eo_x, (int)eo_y));
            ir_mkpts.push_back(cv::Point2i((int)ir_x, (int)ir_y));
        }
        
        std::cout << "Extracted " << len << " feature point pairs" << std::endl;
        
        // Log inference time to CSV
        inference_count_++;
        if (csv_file_.is_open()) {
            csv_file_ << inference_count_ << "," << inference_time_seconds << "," << len << std::endl;
            csv_file_.flush();  // Ensure immediate write to file
        }
    }

    // alias for pred_cpu
    void pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts) {
        pred_cpu(eo, ir, eo_mkpts, ir_mkpts);
    }

    // align with last H
    void align(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, cv::Mat &H) {
        // predict keypoints
        pred(eo, ir, eo_pts, ir_pts);

        std::vector<cv::Point2i> q_diff;
        std::vector<std::vector<int>> q_idx;

        std::vector<int> filter_idx;
        std::vector<float> v_line, h_line;

        // Calculate homography if we have enough points
        if (eo_pts.size() >= 4 && ir_pts.size() >= 4) {
            // Convert to Point2f for homography calculation
            std::vector<cv::Point2f> eo_pts_f, ir_pts_f;
            for (const auto& pt : eo_pts) {
                eo_pts_f.emplace_back(pt.x, pt.y);
            }
            for (const auto& pt : ir_pts) {
                ir_pts_f.emplace_back(pt.x, pt.y);
            }
            
            // Calculate homography using RANSAC
            cv::Mat mask;
            H = cv::findHomography(eo_pts_f, ir_pts_f, cv::RANSAC,  8.0, mask, 800, 0.98);
            
            if (!H.empty() && cv::determinant(H) > 1e-6) {
                std::cout << "Successfully computed homography matrix" << std::endl;
                
                // Filter points based on homography
                std::vector<cv::Point2i> filtered_eo_pts, filtered_ir_pts;
                for (size_t i = 0; i < eo_pts.size(); i++) {
                    if (i < mask.rows && mask.at<uchar>(i) == 1) {
                        filtered_eo_pts.push_back(eo_pts[i]);
                        filtered_ir_pts.push_back(ir_pts[i]);
                    }
                }
                eo_pts = filtered_eo_pts;
                ir_pts = filtered_ir_pts;
                
                std::cout << "Final feature points after homography filtering: " << eo_pts.size() << std::endl;
            } else {
                std::cout << "Failed to compute valid homography matrix" << std::endl;
                H = cv::Mat::eye(3, 3, CV_64F); // Identity matrix as fallback
            }
        } else {
            std::cout << "Insufficient points for homography calculation (" << eo_pts.size() << " points)" << std::endl;
            H = cv::Mat::eye(3, 3, CV_64F); // Identity matrix as fallback
        }
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
};

ImageAlignONNX::ptr ImageAlignONNX::create_instance(const Param& param) {
    return std::make_shared<ImageAlignONNXImpl>(param);
}

}

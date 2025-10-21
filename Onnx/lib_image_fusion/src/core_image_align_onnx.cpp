#include "../include/core_image_align_onnx.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <onnxruntime_cxx_api.h>

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

    std::ofstream csv_file_;
    int inference_count_ = 0;
    std::string current_image_name_ = "";

    void smart_warmup_onnx() {
        std::cout << "Smart warmup for ONNX providers initialization..." << std::endl;
        cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
        cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
        const auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<cv::Point2i> dummy_eo_mkpts, dummy_ir_mkpts;
        try {
            pred_cpu(eo, ir, dummy_eo_mkpts, dummy_ir_mkpts);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Smart warmup failed: " << e.what() << std::endl;
        }

        std::cout << "Recreating ONNX session to maintain first-inference precision..." << std::endl;
        session_.reset();
        session_ = std::make_unique<Ort::Session>(env_, param_.model_path.c_str(), session_options_);

        input_names_.clear();
        output_names_.clear();
        input_names_ptrs_.clear();
        output_names_ptrs_.clear();
        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_input_nodes = session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_ptrs_.push_back(std::move(input_name));
            input_names_.push_back(input_names_ptrs_.back().get());
        }

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

        std::cout << "debug: Setting deterministic seeds for ONNX inference..." << std::endl;
        std::srand(1);
        srand(1);
        std::cout << "debug: Random seeds and environment configured for deterministic ONNX inference (seed=1, matching LibTorch C++)" << std::endl;

        csv_file_.open("onnx_inference_times.csv", std::ios::app);
        if (!csv_file_.is_open()) {
            std::cerr << "Warning: Could not open CSV file for writing inference times" << std::endl;
        } else {

            csv_file_.seekp(0, std::ios::end);
            if (csv_file_.tellp() == 0) {
                csv_file_ << "Image_Name,Inference_Time_Seconds,Features_Count" << std::endl;
            }
        }

        if (!std::experimental::filesystem::exists(param_.model_path)) {
            std::cerr << "FATAL ERROR: Model file not found: " << param_.model_path << std::endl;
            throw std::runtime_error("ONNX model file not found: " + param_.model_path);
        }
        try {

            std::cout << "debug: Configuring ONNX Runtime for FP32 precision and determinism..." << std::endl;

            session_options_.SetIntraOpNumThreads(1);                    
            session_options_.SetInterOpNumThreads(1);                    

            session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

            session_options_.SetLogSeverityLevel(3);

            try {
                session_options_.DisableMemPattern();
            } catch (...) {
                std::cout << "debug: DisableMemPattern not available in this ONNX Runtime version" << std::endl;
            }
            try {
                session_options_.DisableCpuMemArena();
            } catch (...) {
                std::cout << "debug: DisableCpuMemArena not available in this ONNX Runtime version" << std::endl;
            }

            std::cout << "debug: Attempting to initialize ONNX Runtime with device: " << param_.device << std::endl;
            bool use_cuda = false;
            if (param_.device == "cuda") {
                std::cout << "debug: Adding CUDA execution provider with FP32 and deterministic settings..." << std::endl;

                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;

                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;  
                cuda_options.do_copy_in_default_stream = 1;

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

            memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

            Ort::AllocatorWithDefaultOptions allocator;

            size_t num_input_nodes = session_->GetInputCount();
            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = session_->GetInputNameAllocated(i, allocator);
                input_names_ptrs_.push_back(std::move(input_name));
                input_names_.push_back(input_names_ptrs_.back().get());
            }

            size_t num_output_nodes = session_->GetOutputCount();
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = session_->GetOutputNameAllocated(i, allocator);
                output_names_ptrs_.push_back(std::move(output_name));
                output_names_.push_back(output_names_ptrs_.back().get());
            }
            std::cout << "Successfully loaded ONNX model with " << num_input_nodes << " inputs and " << num_output_nodes << " outputs" << std::endl;

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

    ~ImageAlignONNXImpl() {
        if (csv_file_.is_open()) {
            csv_file_.close();
            std::cout << "CSV file with ONNX inference times closed. Total inferences: " << inference_count_ << std::endl;
        }
    }

    void pred_cpu(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts) {
        double inference_time_seconds = 0.0;

        if (eo.channels() != 1 || ir.channels() != 1) {
            throw std::runtime_error("ImageAlignONNXImpl::pred: eo and ir must be single channel images");
        }

        bool use_fp16 = (param_.pred_mode == "fp16");
        std::cout << "debug: Using " << (use_fp16 ? "FP16" : "FP32") << " input (pred_mode=" << param_.pred_mode << ")" << std::endl;
        cv::Mat eo_float, ir_float;
        eo.convertTo(eo_float, CV_32F, 1.0 / 255.0, 0.0);
        ir.convertTo(ir_float, CV_32F, 1.0 / 255.0, 0.0);

        size_t input_size = param_.pred_height * param_.pred_width;

        std::vector<int64_t> input_shape = {1, 1, param_.pred_height, param_.pred_width};

        std::vector<Ort::Value> inputs;
        if (use_fp16) {

            std::vector<Ort::Float16_t> eo_fp16_data(input_size);
            std::vector<Ort::Float16_t> ir_fp16_data(input_size);

            const float* eo_ptr = eo_float.ptr<float>();
            const float* ir_ptr = ir_float.ptr<float>();
            for (size_t i = 0; i < input_size; i++) {
                eo_fp16_data[i] = Ort::Float16_t(eo_ptr[i]);
                ir_fp16_data[i] = Ort::Float16_t(ir_ptr[i]);
            }
            std::cout << "debug: Converted input data to FP16 format" << std::endl;

            static std::vector<Ort::Float16_t> eo_fp16_static, ir_fp16_static;
            eo_fp16_static = std::move(eo_fp16_data);
            ir_fp16_static = std::move(ir_fp16_data);
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                *memory_info_, eo_fp16_static.data(), input_size, input_shape.data(), 4));
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                *memory_info_, ir_fp16_static.data(), input_size, input_shape.data(), 4));
        } else {
            std::cout << "777777777777777777777777777777777777" << std::endl;

            std::vector<float> eo_float32_data(input_size);
            std::vector<float> ir_float32_data(input_size);

            for (size_t i = 0; i < input_size; i++) {
                eo_float32_data[i] = eo_float.ptr<float>()[i];
                ir_float32_data[i] = ir_float.ptr<float>()[i];
            }

            static std::vector<float> eo_fp32_static, ir_fp32_static;
            eo_fp32_static = std::move(eo_float32_data);
            ir_fp32_static = std::move(ir_float32_data);
            inputs.push_back(Ort::Value::CreateTensor<float>(
                *memory_info_, eo_fp32_static.data(), input_size, input_shape.data(), 4));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                *memory_info_, ir_fp32_static.data(), input_size, input_shape.data(), 4));
        }

        auto inference_start = std::chrono::high_resolution_clock::now();

        Ort::RunOptions run_options;
        run_options.SetRunLogSeverityLevel(3);
        std::cout << "debug: Running ONNX model inference (pred_mode=" << param_.pred_mode << ")..." << std::endl;

        auto pred = session_->Run(run_options, input_names_.data(), inputs.data(), 2, 
                                output_names_.data(), output_names_.size());

        auto inference_end = std::chrono::high_resolution_clock::now();
        auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
        inference_time_seconds = inference_duration.count() / 1000000.0;
        std::cout << "ONNX Inference time (" << param_.pred_mode << "): " << inference_time_seconds << " seconds" << std::endl;

        const int32_t *eo_res = pred[0].GetTensorMutableData<int32_t>();
        const int32_t *ir_res = pred[1].GetTensorMutableData<int32_t>();

        auto eo_shape = pred[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_points = static_cast<int>(eo_shape[0]);
        eo_mkpts.clear();
        ir_mkpts.clear();

        for (int i = 0, pt = 0; i < num_points; i++, pt += 2) {
            int eo_x = static_cast<int>(eo_res[pt]);
            int eo_y = static_cast<int>(eo_res[pt + 1]);
            int ir_x = static_cast<int>(ir_res[pt]);
            int ir_y = static_cast<int>(ir_res[pt + 1]);

            if (eo_x == 0 && eo_y == 0) {
                continue;
            }
            eo_mkpts.push_back(cv::Point2i(eo_x, eo_y));
            ir_mkpts.push_back(cv::Point2i(ir_x, ir_y));
        }
        std::cout << "Extracted " << eo_mkpts.size() << " valid feature point pairs (pred_mode=" << param_.pred_mode << ")" << std::endl;

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

    void pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts) {
        pred_cpu(eo, ir, eo_mkpts, ir_mkpts);
    }

    void align(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, cv::Mat &H) {

        pred(eo, ir, eo_pts, ir_pts);

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

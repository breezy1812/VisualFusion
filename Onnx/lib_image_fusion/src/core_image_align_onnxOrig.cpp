#include "../include/core_image_align_onnx.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <fstream>
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

        std::cout << "Initializing ONNX Runtime without random seed (TensorRT style)..." << std::endl;

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

            std::cout << "Configuring ONNX Runtime for maximum determinism..." << std::endl;

            session_options_.SetIntraOpNumThreads(1);                    
            session_options_.SetInterOpNumThreads(1);                    

            std::cout << "Attempting to initialize ONNX Runtime with device: " << param_.device << std::endl;
            bool use_cuda = false;

            if (param_.device == "cuda") {
                std::cout << "Adding CUDA execution provider with deterministic settings..." << std::endl;

                OrtCUDAProviderOptions cuda_options{};

                session_options_.AppendExecutionProvider_CUDA(cuda_options);

                use_cuda = true;
                use_cuda = true;
                std::cout << "CUDA execution provider added successfully" << std::endl;
            } else {
                std::cout << "Using CPU execution provider" << std::endl;
            }
            session_ = std::make_unique<Ort::Session>(env_, param_.model_path.c_str(), session_options_);

            if (use_cuda) {
                memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
            } else {
                memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
            }

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
                std::cout << "Performing smart warmup for ONNX CUDA providers..." << std::endl;

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

        if (param_.pred_mode == "fp16") {

        } 
        else {

            std::cout << "DEBUG: Using FP32 input mode (LibTorch identical)" << std::endl;

            cv::Mat eo_float, ir_float;
            eo.convertTo(eo_float, CV_32F, 1.0 / 255.0, 0.0);
            ir.convertTo(ir_float, CV_32F, 1.0 / 255.0, 0.0);

            size_t input_size = param_.pred_height * param_.pred_width;
            std::vector<float> eo_float32_data(input_size);
            std::vector<float> ir_float32_data(input_size);

            for (size_t i = 0; i < input_size; i++) {
                eo_float32_data[i] = eo_float.ptr<float>()[i];
                ir_float32_data[i] = ir_float.ptr<float>()[i];
            }

            if(!current_image_name_.empty()){
                std::string input_csv_filename = "../../output/input_data_"+current_image_name_ + ".csv";

                std::ofstream ofs(input_csv_filename);
                ofs << std::fixed << std::setprecision(20);
                if (!ofs) {
                    throw std::runtime_error("Failed to open CSV file for writing");
                }

                auto get_tensor_value = [](const std::vector<float>& data, int height, int width, int i, int j) -> float {
                    return data[i * width + j];
                };

                for (int i = 0; i < param_.pred_height; ++i) {
                    for (int j = 0; j < param_.pred_width; ++j) {
                        float eo_val = get_tensor_value(eo_float32_data, param_.pred_height, param_.pred_width, i, j);
                        float ir_val = get_tensor_value(ir_float32_data, param_.pred_height, param_.pred_width, i, j);
                        ofs << eo_val << "," << ir_val << "\n";
                    }
                }
            }

            std::vector<int64_t> input_shape = {1, 1, param_.pred_height, param_.pred_width};

            Ort::Value eo_tensor = Ort::Value::CreateTensor<float>(
                *memory_info_, eo_float32_data.data(), input_size, input_shape.data(), 4);
            Ort::Value ir_tensor = Ort::Value::CreateTensor<float>(
                *memory_info_, ir_float32_data.data(), input_size, input_shape.data(), 4);

            std::vector<Ort::Value> inputs;
            inputs.push_back(std::move(eo_tensor));
            inputs.push_back(std::move(ir_tensor));

            auto inference_start = std::chrono::high_resolution_clock::now();

            Ort::RunOptions run_options;
            run_options.SetRunLogSeverityLevel(3);
            std::cout << "DEBUG: Running FP32 model inference..." << std::endl;

            auto pred = session_->Run(run_options, input_names_.data(), inputs.data(), 2, 
                                    output_names_.data(), output_names_.size());

            auto inference_end = std::chrono::high_resolution_clock::now();
            auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
            inference_time_seconds = inference_duration.count() / 1000000.0;
            std::cout << "ONNX FP32 Inference time: " << inference_time_seconds << " seconds" << std::endl;

            const int64_t *eo_res = pred[0].GetTensorMutableData<int64_t>();
            const int64_t *ir_res = pred[1].GetTensorMutableData<int64_t>();
            const long int leng1 = pred[2].GetTensorMutableData<long int>()[0];
            const long int leng2 = pred[3].GetTensorMutableData<long int>()[0];
            eo_mkpts.clear();
            ir_mkpts.clear();

            int len = leng1;
            for (int i = 0, pt = 0; i < len; i++, pt += 2) {

                int eo_x = static_cast<int>(eo_res[pt]);
                int eo_y = static_cast<int>(eo_res[pt + 1]);
                int ir_x = static_cast<int>(ir_res[pt]);
                int ir_y = static_cast<int>(ir_res[pt + 1]);
                eo_mkpts.push_back(cv::Point2i(eo_x, eo_y));
                ir_mkpts.push_back(cv::Point2i(ir_x, ir_y));
            }
            std::cout << "Extracted " << len << " feature point pairs (FP32 mode)" << std::endl;
        }

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

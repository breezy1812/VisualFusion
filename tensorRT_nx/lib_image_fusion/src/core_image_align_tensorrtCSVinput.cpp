#include "../include/core_image_align_tensorrt.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <cuda_fp16.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "debug:" << msg << std::endl;
        }
    }
};

namespace core {

class ImageAlignTensorRTImpl : public ImageAlignTensorRT {
private:
    Param param_;
    Logger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;

    std::ofstream csv_file_;
    std::string current_image_name_ = "";

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

    ImageAlignTensorRTImpl(const Param& param) : ImageAlignTensorRT(param), param_(param) {
        std::cout << "debug: Initializing ImageAlignTensorRT with pred_mode=" << param_.pred_mode << "..." << std::endl;

        csv_file_.open("tensorrt_inference_times.csv", std::ios::app);
        if (csv_file_.is_open()) {

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

        std::cout << "debug: Performing smart warmup for TensorRT CUDA execution..." << std::endl;
        smart_warmup_tensorrt();
    }

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

    void smart_warmup_tensorrt() {
        std::cout<<"******************************"<<std::endl;
        std::cout << "debug: TensorRT Smart warmup for CUDA kernel initialization..." << std::endl;
        std::cout<<"******************************"<<std::endl;

        cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
        cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;

        const auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<cv::Point2i> dummy_eo_mkpts, dummy_ir_mkpts;
        pred(eo, ir, dummy_eo_mkpts, dummy_ir_mkpts);

        std::cout << "debug: Recreating TensorRT execution context to maintain first-inference precision..." << std::endl;
        if (context_) context_->destroy();
        context_ = engine_->createExecutionContext();
        if (!context_) {
            throw std::runtime_error("Failed to recreate TensorRT execution context");
        }

        const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
        const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

        std::cout << "debug: TensorRT Smart warmup completed in " << std::fixed << std::setprecision(2) << period << " s" << std::endl;
    }

    void set_current_image_name(const std::string& image_name) {
        current_image_name_ = image_name;
    }

    void align(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_pts, std::vector<cv::Point2i>& ir_pts, cv::Mat& H) override {
        std::cout << "debug: Starting alignment process for image: " << current_image_name_ << std::endl;
        pred(eo, ir, eo_pts, ir_pts);

        if (eo_pts.size() < 4) {
            std::cout << "debug: Not enough points to compute homography. Found only " << eo_pts.size() << " points." << std::endl;
            H = cv::Mat::eye(3, 3, CV_64F);
            return;
        }

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

        H = cv::Mat::eye(3, 3, CV_64F);
        std::cout << "debug: Feature point extraction complete. Found " << eo_pts.size() << " points." << std::endl;
    }

    void pred(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_mkpts, std::vector<cv::Point2i>& ir_mkpts) {
        std::cout << "debug: Starting prediction..." << std::endl;

        cv::Mat eo_resized, ir_resized;
        cv::resize(eo, eo_resized, cv::Size(param_.pred_width, param_.pred_height));
        cv::resize(ir, ir_resized, cv::Size(param_.pred_width, param_.pred_height));

        cv::Mat eo_gray, ir_gray;

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

        cv::Mat eo_uint8, ir_uint8;
        eo_gray.convertTo(eo_uint8, CV_8U);
        ir_gray.convertTo(ir_uint8, CV_8U);
        cv::Mat eo_float, ir_float;

        std::cout << "debug: Converting images to FP32 format (unified input)..." << std::endl;
        eo_uint8.convertTo(eo_float, CV_32F, 1.0f / 255.0f);
        ir_uint8.convertTo(ir_float, CV_32F, 1.0f / 255.0f);
        if (param_.pred_mode == "fp16") {
            std::cout << "debug: Using FP16 TensorRT engine with FP32 input data..." << std::endl;
        } else {
            std::cout << "debug: Using FP32 TensorRT engine with FP32 input data..." << std::endl;
        }

        int valid_points_count = 0;
        std::vector<float> eo_data(param_.pred_width * param_.pred_height);
        std::vector<float> ir_data(param_.pred_width * param_.pred_height);

        memcpy(eo_data.data(), eo_float.data, eo_data.size() * sizeof(float));
        memcpy(ir_data.data(), ir_float.data, ir_data.size() * sizeof(float));
        std::cout << "debug: Pre-processing complete. Using FP32 input data. Image size: " 
                  << param_.pred_width << "x" << param_.pred_height << std::endl;

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

        if (valid_points_count > 0 && valid_points_count <= (int)eo_mkpts.size()) {

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

    bool runInference(const std::vector<float>& eo_data, const std::vector<float>& ir_data, 
                      std::vector<cv::Point2i>& eo_kps, std::vector<cv::Point2i>& ir_kps, int& leng1) {
        const int num_bindings = engine_->getNbBindings();
        std::cout << "debug: [runInference] Total bindings: " << num_bindings << std::endl;

        std::vector<void*> buffers(num_bindings);

        for (int i = 0; i < num_bindings; ++i) {
            auto dims = engine_->getBindingDimensions(i);
            const char* binding_name = engine_->getBindingName(i);
            nvinfer1::DataType dtype = engine_->getBindingDataType(i);
            size_t element_size = 0;
            std::string type_str = "UNKNOWN";

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
                    std::cerr << "debug: ERROR: Unsupported data type for binding " << binding_name << " (Type: " << static_cast<int>(dtype) << ")" << std::endl;

                    for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                    return false;
            }
            std::cout << "debug: [runInference] Binding: " << i << ", Name: " << binding_name << ", Type: " << type_str;
            std::cout << ", Shape: [";
            for (int d = 0; d < dims.nbDims; ++d) {
                std::cout << dims.d[d];
                if (d < dims.nbDims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>()) * element_size;
            if (cudaMalloc(&buffers[i], size) != cudaSuccess) {
                std::cerr << "debug: ERROR: CUDA memory allocation failed for binding " << i << " (" << binding_name << ")" << std::endl;

                for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                return false;
            }
        }

        int eo_img_idx = engine_->getBindingIndex("vi_img");
        int ir_img_idx = engine_->getBindingIndex("ir_img");
        if (eo_img_idx < 0 || ir_img_idx < 0) {
            std::cerr << "debug: ERROR: Could not find input bindings vi_img or ir_img" << std::endl;

            for(void* buf : buffers) cudaFree(buf);
            return false;
        }

        cudaMemcpyAsync(buffers[eo_img_idx], eo_data.data(), eo_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(buffers[ir_img_idx], ir_data.data(), ir_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);

        std::cout << "debug: Executing model..." << std::endl;
        if (!context_->enqueueV2(buffers.data(), stream_, nullptr)) {
            std::cerr << "debug: ERROR: Failed to enqueue inference." << std::endl;
            for(void* buf : buffers) cudaFree(buf);
            return false;
        }

        cudaStreamSynchronize(stream_);
        std::cout << "debug: Model execution complete." << std::endl;

        int output_count = 0;
        std::vector<int> output_indices;
        std::vector<std::string> output_names;

        for (int i = 0; i < num_bindings; ++i) {
            if (!engine_->bindingIsInput(i)) {
                output_indices.push_back(i);
                output_names.push_back(engine_->getBindingName(i));
                output_count++;
            }
        }
        std::cout << "debug: Found " << output_count << " output bindings:" << std::endl;
        for (size_t i = 0; i < output_names.size(); ++i) {
            auto dims = engine_->getBindingDimensions(output_indices[i]);
            std::cout << "  Output " << i << ": " << output_names[i] << " (binding " << output_indices[i] << "), shape: [";
            for (int d = 0; d < dims.nbDims; ++d) {
                std::cout << dims.d[d];
                if (d < dims.nbDims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        if (output_count < 2) {
            std::cerr << "debug: ERROR: Expected at least 2 feature tensor outputs, got " << output_count << std::endl;
            for(void* buf : buffers) cudaFree(buf);
            return false;
        }

        int feat_vi_idx = output_indices[0];  
        int feat_ir_idx = output_indices[1];
        auto feat_vi_dims = engine_->getBindingDimensions(feat_vi_idx);
        auto feat_ir_dims = engine_->getBindingDimensions(feat_ir_idx);

        size_t feat_vi_size = std::accumulate(feat_vi_dims.d, feat_vi_dims.d + feat_vi_dims.nbDims, 1, std::multiplies<size_t>());
        size_t feat_ir_size = std::accumulate(feat_ir_dims.d, feat_ir_dims.d + feat_ir_dims.nbDims, 1, std::multiplies<size_t>());
        std::cout << "debug: feat_vi tensor size: " << feat_vi_size << " elements" << std::endl;
        std::cout << "debug: feat_ir tensor size: " << feat_ir_size << " elements" << std::endl;

        std::vector<float> feat_vi_data(feat_vi_size);
        std::vector<float> feat_ir_data(feat_ir_size);

        cudaMemcpy(feat_vi_data.data(), buffers[feat_vi_idx], feat_vi_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(feat_ir_data.data(), buffers[feat_ir_idx], feat_ir_size * sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "debug: Feature tensors copied to host memory" << std::endl;

        std::cout << "debug: First 10 feat_vi values: ";
        for (int i = 0; i < std::min(10, (int)feat_vi_size); ++i) {
            std::cout << std::fixed << std::setprecision(6) << feat_vi_data[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "debug: First 10 feat_ir values: ";
        for (int i = 0; i < std::min(10, (int)feat_ir_size); ++i) {
            std::cout << std::fixed << std::setprecision(6) << feat_ir_data[i] << " ";
        }
        std::cout << std::endl;

        if (!current_image_name_.empty()) {

            system("mkdir -p ../../../tensorRT/output");
            std::string csv_filename = "../../../tensorRT/output/feat_data_" + current_image_name_ + ".csv";
            std::ofstream csv_file(csv_filename);
            csv_file << std::fixed << std::setprecision(20);
            if (csv_file.is_open()) {
                std::cout << "debug: Saving TensorRT feature data to CSV: " << csv_filename << std::endl;

                size_t min_size = std::min(feat_vi_size, feat_ir_size);
                for (size_t i = 0; i < min_size; ++i) {
                    float vi_val = feat_vi_data[i];
                    float ir_val = feat_ir_data[i];
                    csv_file << vi_val << "," << ir_val << "\n";
                }
                csv_file.close();
                std::cout << "debug: Saved " << min_size << " feature values to CSV" << std::endl;
            } else {
                std::cerr << "debug: ERROR: Could not create CSV file: " << csv_filename << std::endl;
            }
        }

        eo_kps.clear();
        ir_kps.clear();
        leng1 = 0;

        for (void* buf : buffers) {
            cudaFree(buf);
        }

        std::cout << "debug: [runInference] Feature tensor processing completed" << std::endl;
        return true;
    }

    bool runInferenceFP16(const std::vector<__half>& eo_data, const std::vector<__half>& ir_data, 
                         std::vector<cv::Point2i>& eo_kps, std::vector<cv::Point2i>& ir_kps, int& leng1) {
        const int num_bindings = engine_->getNbBindings();
        std::cout << "debug: [runInferenceFP16] Total bindings: " << num_bindings << std::endl;

        std::vector<void*> buffers(num_bindings);

        for (int i = 0; i < num_bindings; ++i) {
            auto dims = engine_->getBindingDimensions(i);
            const char* binding_name = engine_->getBindingName(i);
            nvinfer1::DataType dtype = engine_->getBindingDataType(i);
            size_t element_size = 0;
            std::string type_str = "UNKNOWN";

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

                    for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                    return false;
            }
            std::cout << "debug: [runInferenceFP16] Binding: " << i << ", Name: " << binding_name << ", Type: " << type_str;
            std::cout << ", Shape: [";
            for (int d = 0; d < dims.nbDims; ++d) {
                std::cout << dims.d[d];
                if (d < dims.nbDims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>()) * element_size;
            if (cudaMalloc(&buffers[i], size) != cudaSuccess) {
                std::cerr << "debug: ERROR: CUDA memory allocation failed for FP16 binding " << i << " (" << binding_name << ")" << std::endl;

                for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                return false;
            }
        }

        int eo_img_idx = engine_->getBindingIndex("vi_img");
        int ir_img_idx = engine_->getBindingIndex("ir_img");

        if (eo_img_idx < 0 || ir_img_idx < 0) {
            std::cerr << "debug: ERROR: Could not find input bindings vi_img or ir_img for FP16" << std::endl;

            for(void* buf : buffers) cudaFree(buf);
            return false;
        }

        cudaMemcpyAsync(buffers[eo_img_idx], eo_data.data(), eo_data.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(buffers[ir_img_idx], ir_data.data(), ir_data.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);

        std::cout << "debug: Executing FP16 model..." << std::endl;
        if (!context_->enqueueV2(buffers.data(), stream_, nullptr)) {
            std::cerr << "debug: ERROR: Failed to enqueue FP16 inference." << std::endl;
            for(void* buf : buffers) cudaFree(buf);
            return false;
        }

        cudaStreamSynchronize(stream_);
        std::cout << "debug: FP16 model execution complete." << std::endl;

        int output_count = 0;
        std::vector<int> output_indices;
        std::vector<std::string> output_names;
        for (int i = 0; i < num_bindings; ++i) {
            if (!engine_->bindingIsInput(i)) {
                output_indices.push_back(i);
                output_names.push_back(engine_->getBindingName(i));
                output_count++;
            }
        }
        std::cout << "debug: Found " << output_count << " FP16 output bindings:" << std::endl;
        for (size_t i = 0; i < output_names.size(); ++i) {
            auto dims = engine_->getBindingDimensions(output_indices[i]);
            std::cout << "  Output " << i << ": " << output_names[i] << " (binding " << output_indices[i] << "), shape: [";
            for (int d = 0; d < dims.nbDims; ++d) {
                std::cout << dims.d[d];
                if (d < dims.nbDims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        if (output_count < 2) {
            std::cerr << "debug: ERROR: Expected at least 2 feature tensor outputs for FP16, got " << output_count << std::endl;
            for(void* buf : buffers) cudaFree(buf);
            return false;
        }

        int feat_vi_idx = output_indices[0];  
        int feat_ir_idx = output_indices[1];
        auto feat_vi_dims = engine_->getBindingDimensions(feat_vi_idx);
        auto feat_ir_dims = engine_->getBindingDimensions(feat_ir_idx);
        nvinfer1::DataType feat_vi_dtype = engine_->getBindingDataType(feat_vi_idx);
        nvinfer1::DataType feat_ir_dtype = engine_->getBindingDataType(feat_ir_idx);

        size_t feat_vi_size = std::accumulate(feat_vi_dims.d, feat_vi_dims.d + feat_vi_dims.nbDims, 1, std::multiplies<size_t>());
        size_t feat_ir_size = std::accumulate(feat_ir_dims.d, feat_ir_dims.d + feat_ir_dims.nbDims, 1, std::multiplies<size_t>());
        std::cout << "debug: FP16 feat_vi tensor size: " << feat_vi_size << " elements, dtype: " << static_cast<int>(feat_vi_dtype) << std::endl;
        std::cout << "debug: FP16 feat_ir tensor size: " << feat_ir_size << " elements, dtype: " << static_cast<int>(feat_ir_dtype) << std::endl;

        std::vector<float> feat_vi_data(feat_vi_size);
        std::vector<float> feat_ir_data(feat_ir_size);
        if (feat_vi_dtype == nvinfer1::DataType::kHALF) {

            std::vector<__half> feat_vi_fp16(feat_vi_size);
            std::vector<__half> feat_ir_fp16(feat_ir_size);
            cudaMemcpy(feat_vi_fp16.data(), buffers[feat_vi_idx], feat_vi_size * sizeof(__half), cudaMemcpyDeviceToHost);
            cudaMemcpy(feat_ir_fp16.data(), buffers[feat_ir_idx], feat_ir_size * sizeof(__half), cudaMemcpyDeviceToHost);

            for (size_t i = 0; i < feat_vi_size; ++i) {
                feat_vi_data[i] = static_cast<float>(feat_vi_fp16[i]);
            }
            for (size_t i = 0; i < feat_ir_size; ++i) {
                feat_ir_data[i] = static_cast<float>(feat_ir_fp16[i]);
            }
        } else {

            cudaMemcpy(feat_vi_data.data(), buffers[feat_vi_idx], feat_vi_size * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(feat_ir_data.data(), buffers[feat_ir_idx], feat_ir_size * sizeof(float), cudaMemcpyDeviceToHost);
        }
        std::cout << "debug: FP16 feature tensors copied to host memory" << std::endl;

        std::cout << "debug: First 10 FP16 feat_vi values: ";
        for (int i = 0; i < std::min(10, (int)feat_vi_size); ++i) {
            std::cout << std::fixed << std::setprecision(6) << feat_vi_data[i] << " ";
        }
        std::cout << std::endl;

        if (!current_image_name_.empty()) {

            system("mkdir -p ../../../tensorRT/output");
            std::string csv_filename = "../../../tensorRT/output/feat_data_fp16_" + current_image_name_ + ".csv";
            std::ofstream csv_file(csv_filename);
            csv_file << std::fixed << std::setprecision(20);
            if (csv_file.is_open()) {
                std::cout << "debug: Saving TensorRT FP16 feature data to CSV: " << csv_filename << std::endl;

                size_t min_size = std::min(feat_vi_size, feat_ir_size);
                for (size_t i = 0; i < min_size; ++i) {
                    float vi_val = feat_vi_data[i];
                    float ir_val = feat_ir_data[i];
                    csv_file << vi_val << "," << ir_val << "\n";
                }
                csv_file.close();
                std::cout << "debug: Saved " << min_size << " FP16 feature values to CSV" << std::endl;
            } else {
                std::cerr << "debug: ERROR: Could not create FP16 CSV file: " << csv_filename << std::endl;
            }
        }

        eo_kps.clear();
        ir_kps.clear();
        leng1 = 0;

        for (void* buf : buffers) {
            cudaFree(buf);
        }

        std::cout << "debug: [runInferenceFP16] Feature tensor processing completed" << std::endl;
        return true;
    }
};

std::shared_ptr<ImageAlignTensorRT> ImageAlignTensorRT::create_instance(const Param& param) {
    return std::make_shared<ImageAlignTensorRTImpl>(param);
}

ImageAlignTensorRT::ImageAlignTensorRT(const Param& param) {}
ImageAlignTensorRT::~ImageAlignTensorRT() {}

}

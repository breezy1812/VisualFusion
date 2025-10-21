

#include "../include/core_image_align_tensorrt.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <experimental/filesystem>
#include <cuda_fp16.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {

        if (severity <= Severity::kERROR) {
            std::cout << msg << std::endl;
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
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open engine file: " << engine_path << std::endl;
            return false;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            std::cerr << "ERROR: Could not read engine file." << std::endl;
            return false;
        }

        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_) {
            std::cerr << "ERROR: Failed to create TensorRT runtime." << std::endl;
            return false;
        }

        engine_ = runtime_->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
        if (!engine_) {
            std::cerr << "ERROR: Failed to deserialize CUDA engine." << std::endl;
            return false;
        }
        return true;
    }

public:

    ImageAlignTensorRTImpl(const Param& param) : ImageAlignTensorRT(param), param_(param) {

        setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
        printf("Model initialization completed\n");

        std::string csv_filename = "./itiming_log.csv";
        bool file_exists = std::experimental::filesystem::exists(csv_filename);
        csv_file_.open(csv_filename, std::ios::app);
        if (!file_exists && csv_file_.is_open()) {

            csv_file_ << "Filename,Operation,Time_s,Mode,keypoints\n";
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

        printf("Performing smart warmup to initialize CUDA kernels...\n");
        smart_warmup_tensorrt();
    }

    ~ImageAlignTensorRTImpl() {
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
        if (stream_) cudaStreamDestroy(stream_);
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
        if (runtime_) runtime_->destroy();
    }

    void smart_warmup_tensorrt() {
        printf("Smart warmup for CUDA kernel initialization...\n");

        cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
        cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;

        const auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<cv::Point2i> dummy_eo_mkpts, dummy_ir_mkpts;
        pred(eo, ir, dummy_eo_mkpts, dummy_ir_mkpts);

        printf("Recreating TensorRT execution context to maintain first-inference precision...\n");
        if (context_) context_->destroy();
        context_ = engine_->createExecutionContext();
        if (!context_) {
            throw std::runtime_error("Failed to recreate TensorRT execution context");
        }

        const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
        const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

        printf("Smart warmup completed in %.2f s\n", period);
    }

    void set_current_image_name(const std::string& image_name) {
        current_image_name_ = image_name;
    }

    void align(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_pts, std::vector<cv::Point2i>& ir_pts, cv::Mat& H) override {

        pred(eo, ir, eo_pts, ir_pts);

        if (std::abs(param_.out_width_scale - 1.0) > 1e-6 || std::abs(param_.out_height_scale - 1.0) > 1e-6 || param_.bias_x > 0 || param_.bias_y > 0)
        {
            for (cv::Point2i &i : eo_pts)
            {
                i.x = i.x * param_.out_width_scale;
                i.y = i.y * param_.out_height_scale;
            }
            for (cv::Point2i &i : ir_pts)
            {
                i.x = i.x * param_.out_width_scale;
                i.y = i.y * param_.out_height_scale;
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

        std::cout << "  - Final feature points after coordinate adjustment: " << eo_pts.size() << std::endl;
    }

    void pred(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_mkpts, std::vector<cv::Point2i>& ir_mkpts) {

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
        eo_gray.convertTo(eo_float, CV_32F, 1.0 / 255.0, 0.0);
        ir_gray.convertTo(ir_float, CV_32F, 1.0 / 255.0, 0.0);

        bool use_fp16 = (param_.pred_mode == "fp16");
        bool use_int8 = (param_.pred_mode == "int8");
        std::string mode_str = use_int8 ? "INT8" : (use_fp16 ? "FP16" : "FP32");
        std::cout << "debug: Using " << mode_str << " input (pred_mode=" << param_.pred_mode << ")" << std::endl;
        int leng = 0;
        bool success = false;

        std::vector<float> eo_data(param_.pred_width * param_.pred_height);
        std::vector<float> ir_data(param_.pred_width * param_.pred_height);

        memcpy(eo_data.data(), eo_float.data, eo_data.size() * sizeof(float));
        memcpy(ir_data.data(), ir_float.data, ir_data.size() * sizeof(float));
        auto model_inference_start = std::chrono::high_resolution_clock::now();

        success = runInference(eo_data, ir_data, eo_mkpts, ir_mkpts, leng);

        auto model_inference_end = std::chrono::high_resolution_clock::now();
        double model_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(model_inference_end - model_inference_start).count() / 1000000.0;
        if (!success) {
            eo_mkpts.clear();
            ir_mkpts.clear();
            return;
        }

        printf("[DEBUG] leng from model: %d, eo_mkpts.size(): %lu\n", leng, eo_mkpts.size());

        writeTimingToCSV("Model_Inference", model_inference_time, leng, current_image_name_);
        printf("Model inference time: %.6f s\n", model_inference_time);

        std::cout << "  - Model extracted " << eo_mkpts.size() << " feature point pairs" << std::endl;
    }

    bool runInference(const std::vector<float>& eo_data, const std::vector<float>& ir_data, 
                      std::vector<cv::Point2i>& eo_kps, std::vector<cv::Point2i>& ir_kps, int& leng1) {
        const int num_bindings = engine_->getNbBindings();

        if (num_bindings != 4) {
            std::cerr << "ERROR: Expected 4 bindings (2 inputs + 2 outputs), but got " << num_bindings << std::endl;
            return false;
        }

        std::vector<void*> buffers(num_bindings);

        for (int i = 0; i < num_bindings; ++i) {
            auto dims = engine_->getBindingDimensions(i);
            nvinfer1::DataType dtype = engine_->getBindingDataType(i);
            size_t element_size = 0;

            switch (dtype) {
                case nvinfer1::DataType::kFLOAT: 
                    element_size = sizeof(float); 
                    break;
                case nvinfer1::DataType::kHALF:
                    element_size = sizeof(__half); 
                    break;
                case nvinfer1::DataType::kINT32: 
                    element_size = sizeof(int32_t); 
                    break;
                case nvinfer1::DataType::kINT8:
                    element_size = sizeof(int8_t); 
                    break;
                default: 
                    std::cerr << "ERROR: Unsupported data type at binding " << i << std::endl;
                    for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                    return false;
            }
            size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>()) * element_size;
            if (cudaMalloc(&buffers[i], size) != cudaSuccess) {
                for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                return false;
            }
        }

        int eo_img_idx = engine_->getBindingIndex("vi_img");
        int ir_img_idx = engine_->getBindingIndex("ir_img");
        int mkpt0_idx = engine_->getBindingIndex("mkpt0");
        int mkpt1_idx = engine_->getBindingIndex("mkpt1");

        if (eo_img_idx < 0 || ir_img_idx < 0 || mkpt0_idx < 0 || mkpt1_idx < 0) {
            std::cerr << "ERROR: Could not find required bindings" << std::endl;
            for(void* buf : buffers) cudaFree(buf);
            return false;
        }

        nvinfer1::DataType input_dtype = engine_->getBindingDataType(eo_img_idx);
        if (input_dtype == nvinfer1::DataType::kHALF) {

            std::cout << "debug: Converting FP32 input to FP16 for TRT FP16 engine" << std::endl;
            std::vector<__half> eo_data_fp16(eo_data.size());
            std::vector<__half> ir_data_fp16(ir_data.size());
            for (size_t i = 0; i < eo_data.size(); i++) {
                eo_data_fp16[i] = __float2half(eo_data[i]);
                ir_data_fp16[i] = __float2half(ir_data[i]);
            }
            cudaMemcpyAsync(buffers[eo_img_idx], eo_data_fp16.data(), eo_data_fp16.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(buffers[ir_img_idx], ir_data_fp16.data(), ir_data_fp16.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);
        } else if (input_dtype == nvinfer1::DataType::kINT8) {

            std::cout << "debug: Converting FP32 input to INT8 for TRT INT8 engine" << std::endl;
            std::vector<int8_t> eo_data_int8(eo_data.size());
            std::vector<int8_t> ir_data_int8(ir_data.size());

            for (size_t i = 0; i < eo_data.size(); i++) {
                eo_data_int8[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, eo_data[i] * 255.0f - 128.0f)));
                ir_data_int8[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, ir_data[i] * 255.0f - 128.0f)));
            }
            cudaMemcpyAsync(buffers[eo_img_idx], eo_data_int8.data(), eo_data_int8.size() * sizeof(int8_t), cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(buffers[ir_img_idx], ir_data_int8.data(), ir_data_int8.size() * sizeof(int8_t), cudaMemcpyHostToDevice, stream_);
        } else {

            cudaMemcpyAsync(buffers[eo_img_idx], eo_data.data(), eo_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(buffers[ir_img_idx], ir_data.data(), ir_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream_);
        }

        if (!context_->enqueueV2(buffers.data(), stream_, nullptr)) {
            return false;
        }

        auto mkpt0_dims = engine_->getBindingDimensions(mkpt0_idx);
        size_t mkpt0_count = std::accumulate(mkpt0_dims.d, mkpt0_dims.d + mkpt0_dims.nbDims, 1, std::multiplies<size_t>());
        std::vector<int32_t> eo_kps_raw(mkpt0_count);

        auto mkpt1_dims = engine_->getBindingDimensions(mkpt1_idx);
        size_t mkpt1_count = std::accumulate(mkpt1_dims.d, mkpt1_dims.d + mkpt1_dims.nbDims, 1, std::multiplies<size_t>());
        std::vector<int32_t> ir_kps_raw(mkpt1_count);

        cudaMemcpyAsync(eo_kps_raw.data(), buffers[mkpt0_idx], eo_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(ir_kps_raw.data(), buffers[mkpt1_idx], ir_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        int num_keypoints = 0;
        int num_coords = 2;

        if (mkpt0_dims.nbDims == 3) {
            num_keypoints = mkpt0_dims.d[1];
            num_coords = mkpt0_dims.d[2];
        } else if (mkpt0_dims.nbDims == 2) {
            num_keypoints = mkpt0_dims.d[0];
            num_coords = mkpt0_dims.d[1];
        } else {
            return false;
        }
        if (num_coords != 2) {
             return false;
        }

        eo_kps.clear();
        ir_kps.clear();

        for (int i = 0; i < num_keypoints; ++i) {
            int x_eo = static_cast<int>(eo_kps_raw[i * num_coords + 0]);
            int y_eo = static_cast<int>(eo_kps_raw[i * num_coords + 1]);
            int x_ir = static_cast<int>(ir_kps_raw[i * num_coords + 0]);
            int y_ir = static_cast<int>(ir_kps_raw[i * num_coords + 1]);

            if (x_eo == 0 && y_eo == 0) {
                continue;
            }
            eo_kps.emplace_back(x_eo, y_eo);
            ir_kps.emplace_back(x_ir, y_ir);
        }
        leng1 = eo_kps.size();

        for (void* buf : buffers) {
            cudaFree(buf);
        }

        return true;
    }

    bool runInferenceFP16(const std::vector<__half>& eo_data, const std::vector<__half>& ir_data, 
                         std::vector<cv::Point2i>& eo_kps, std::vector<cv::Point2i>& ir_kps, int& leng1) {
        const int num_bindings = engine_->getNbBindings();
        if (num_bindings != 4) {
            std::cerr << "debug: ERROR: Expected 4 bindings for FP16, but got " << num_bindings << std::endl;
            return false;
        }

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
            std::cout << "debug: [runInferenceFP16] Binding: " << i << ", Name: " << binding_name << ", Type: " << type_str << std::endl;
            size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>()) * element_size;
            if (cudaMalloc(&buffers[i], size) != cudaSuccess) {
                std::cerr << "debug: ERROR: CUDA memory allocation failed for FP16 binding " << i << " (" << binding_name << ")" << std::endl;

                for(int j = 0; j < i; ++j) cudaFree(buffers[j]);
                return false;
            }
        }

        int eo_img_idx = engine_->getBindingIndex("vi_img");
        int ir_img_idx = engine_->getBindingIndex("ir_img");
        int mkpt0_idx = engine_->getBindingIndex("mkpt0");
        int mkpt1_idx = engine_->getBindingIndex("mkpt1");

        if (eo_img_idx < 0 || ir_img_idx < 0 || mkpt0_idx < 0 || mkpt1_idx < 0) {
            std::cerr << "debug: ERROR: Could not find required bindings for FP16." << std::endl;
            for(void* buf : buffers) cudaFree(buf);
            return false;
        }

        cudaMemcpyAsync(buffers[eo_img_idx], eo_data.data(), eo_data.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(buffers[ir_img_idx], ir_data.data(), ir_data.size() * sizeof(__half), cudaMemcpyHostToDevice, stream_);

        std::cout << "debug: Executing FP16 model..." << std::endl;
        if (!context_->enqueueV2(buffers.data(), stream_, nullptr)) {
            std::cerr << "debug: ERROR: Failed to enqueue FP16 inference." << std::endl;
            return false;
        }

        auto mkpt0_dims = engine_->getBindingDimensions(mkpt0_idx);
        size_t mkpt0_count = std::accumulate(mkpt0_dims.d, mkpt0_dims.d + mkpt0_dims.nbDims, 1, std::multiplies<size_t>());
        std::vector<int32_t> eo_kps_raw(mkpt0_count);

        auto mkpt1_dims = engine_->getBindingDimensions(mkpt1_idx);
        size_t mkpt1_count = std::accumulate(mkpt1_dims.d, mkpt1_dims.d + mkpt1_dims.nbDims, 1, std::multiplies<size_t>());
        std::vector<int32_t> ir_kps_raw(mkpt1_count);

        cudaMemcpyAsync(eo_kps_raw.data(), buffers[mkpt0_idx], eo_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(ir_kps_raw.data(), buffers[mkpt1_idx], ir_kps_raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_);

        cudaStreamSynchronize(stream_);
        std::cout << "debug: FP16 model execution and data copy complete." << std::endl;

        int num_keypoints = 0;
        int num_coords = 2;

        if (mkpt0_dims.nbDims == 3) {
            num_keypoints = mkpt0_dims.d[1];
            num_coords = mkpt0_dims.d[2];
        } else if (mkpt0_dims.nbDims == 2) {
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
        std::cout << "debug: [runInferenceFP16] Parsing all " << num_keypoints << " keypoints, filtering (0,0)..." << std::endl;

        for (int i = 0; i < num_keypoints; ++i) {
            int x_eo = static_cast<int>(std::round(eo_kps_raw[i * num_coords + 0]));
            int y_eo = static_cast<int>(std::round(eo_kps_raw[i * num_coords + 1]));
            int x_ir = static_cast<int>(std::round(ir_kps_raw[i * num_coords + 0]));
            int y_ir = static_cast<int>(std::round(ir_kps_raw[i * num_coords + 1]));

            if (x_eo == 0 && y_eo == 0) {
                continue;
            }
            eo_kps.emplace_back(x_eo, y_eo);
            ir_kps.emplace_back(x_ir, y_ir);
        }
        leng1 = eo_kps.size();
        std::cout << "debug: [runInferenceFP16] After filtering. Valid keypoints: " << eo_kps.size() << std::endl;

        for (int i = 0; i < std::min(5, (int)eo_kps.size()); ++i) {
            std::cout << "debug: [runInferenceFP16] Valid KP " << i << ": EO(" << eo_kps[i].x << "," << eo_kps[i].y 
                      << "), IR(" << ir_kps[i].x << "," << ir_kps[i].y << ")" << std::endl;
        }

        for (void* buf : buffers) {
            cudaFree(buf);
        }

        return true;
    }

    void writeTimingToCSV(const std::string& operation, double time_s, int leng, const std::string& filename)
    {
        std::string csv_filename = "./itiming_log.csv";
        bool file_exists = std::experimental::filesystem::exists(csv_filename);
        std::ofstream csv_file(csv_filename, std::ios::app);
        if (!file_exists) {

            csv_file << "Filename,Operation,Time_s,Mode,keypoints\n";
        }

        std::string identifier;
        if (!filename.empty()) {
            identifier = filename;
        } else {
            std::cout << "warm up" << std::endl;
            return;
        }

        csv_file << identifier << "," 
                 << operation << "," 
                 << std::fixed << std::setprecision(6) << time_s << ","
                 << param_.pred_mode << ","
                 << leng << "\n";
        csv_file.close();
    }
};

std::shared_ptr<ImageAlignTensorRT> ImageAlignTensorRT::create_instance(const Param& param) {
    return std::make_shared<ImageAlignTensorRTImpl>(param);
}

ImageAlignTensorRT::ImageAlignTensorRT(const Param& param) {}
ImageAlignTensorRT::~ImageAlignTensorRT() {}

}

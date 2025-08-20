#include "../include/core_image_align_tensorrt.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstdint>

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
        std::cout << "debug: Initializing ImageAlignTensorRT..." << std::endl;
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
        std::cout << "debug: ImageAlignTensorRT initialized successfully." << std::endl;
    }

    // Destructor
    ~ImageAlignTensorRTImpl() {
        std::cout << "debug: Releasing ImageAlignTensorRT resources..." << std::endl;
        if (stream_) cudaStreamDestroy(stream_);
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
        if (runtime_) runtime_->destroy();
        std::cout << "debug: Resources released." << std::endl;
    }

    // Main alignment function
    void align(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_pts, std::vector<cv::Point2i>& ir_pts, cv::Mat& H) override {
        std::cout << "debug: Starting alignment process..." << std::endl;
        pred(eo, ir, eo_pts, ir_pts);

        if (eo_pts.size() < 4) {
            std::cout << "debug: Not enough points to compute homography. Found only " << eo_pts.size() << " points." << std::endl;
            H = cv::Mat::eye(3, 3, CV_64F);
            return;
        }

        std::vector<cv::Point2f> eo_pts_f, ir_pts_f;
        for(const auto& p : eo_pts) eo_pts_f.push_back(cv::Point2f(p.x, p.y));
        for(const auto& p : ir_pts) ir_pts_f.push_back(cv::Point2f(p.x, p.y));

        cv::Mat mask;
        std::cout << "debug: [align] Attempting to find homography with RANSAC using " << eo_pts_f.size() << " points." << std::endl;
        H = cv::findHomography(eo_pts_f, ir_pts_f, cv::RANSAC, 8.0, mask, 2000, 0.995);

        // Fallback: If RANSAC fails, try again with all points using least-squares method.
        if (H.empty()) {
            std::cout << "debug: [align] RANSAC failed. Retrying with simple least-squares (using all points)." << std::endl;
            H = cv::findHomography(eo_pts_f, ir_pts_f, 0); // method = 0 for least-squares
        }

        if (H.empty() || cv::determinant(H) < 1e-6) {
            std::cout << "debug: Homography computation failed or result is degenerate even after fallback." << std::endl;
            H = cv::Mat::eye(3, 3, CV_64F);
            return;
        }

        // Filter points using the mask from RANSAC, only if RANSAC was successful
        if (!mask.empty()) {
            std::vector<cv::Point2i> inlier_eo_pts, inlier_ir_pts;
            int inlier_count = 0;
            for (int i = 0; i < mask.rows; ++i) {
                if (mask.at<uchar>(i)) {
                    inlier_eo_pts.push_back(eo_pts[i]);
                    inlier_ir_pts.push_back(ir_pts[i]);
                    inlier_count++;
                }
            }
            eo_pts = inlier_eo_pts;
            ir_pts = inlier_ir_pts;
            std::cout << "debug: Homography computed with RANSAC. Inliers: " << inlier_count << "/" << mask.rows << std::endl;
        } else {
            std::cout << "debug: Homography computed with least-squares (all points)." << std::endl;
        }
    }

    // Pre-processing and inference prediction
    void pred(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_mkpts, std::vector<cv::Point2i>& ir_mkpts) {
        std::cout << "debug: Starting prediction..." << std::endl;
        
        // 1. Pre-processing
        cv::Mat eo_resized, ir_resized;
        cv::resize(eo, eo_resized, cv::Size(param_.input_w, param_.input_h));
        cv::resize(ir, ir_resized, cv::Size(param_.input_w, param_.input_h));

        cv::Mat eo_gray, ir_gray;
        cv::cvtColor(eo_resized, eo_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(ir_resized, ir_gray, cv::COLOR_BGR2GRAY);

        cv::Mat eo_float, ir_float;
        eo_gray.convertTo(eo_float, CV_32F, 1.0 / 255.0);
        ir_gray.convertTo(ir_float, CV_32F, 1.0 / 255.0);

        // HWC to CHW
        std::vector<float> eo_data(param_.input_w * param_.input_h);
        std::vector<float> ir_data(param_.input_w * param_.input_h);
        
        // Assuming the model wants [1, C, H, W] where C=1
        memcpy(eo_data.data(), eo_float.data, eo_data.size() * sizeof(float));
        memcpy(ir_data.data(), ir_float.data, ir_data.size() * sizeof(float));
        
        std::cout << "debug: Pre-processing complete. Image size: " << param_.input_w << "x" << param_.input_h << std::endl;

        // 2. Run Inference
        int valid_points_count = 0;
        std::cout << "debug: [pred] Before runInference. eo_mkpts size: " << eo_mkpts.size() << ", ir_mkpts size: " << ir_mkpts.size() << std::endl;
        bool success = runInference(eo_data, ir_data, eo_mkpts, ir_mkpts, valid_points_count);
        std::cout << "debug: [pred] After runInference. eo_mkpts size: " << eo_mkpts.size() << ", ir_mkpts size: " << ir_mkpts.size() << ", valid_points_count: " << valid_points_count << std::endl;

        if (!success) {
            std::cerr << "debug: ERROR: Inference failed." << std::endl;
            eo_mkpts.clear();
            ir_mkpts.clear();
            return;
        }
        
        std::cout << "debug: Inference successful. Total valid points to select: " << valid_points_count << std::endl;

        // 3. Post-processing: Select top 'valid_points_count' points
        std::cout << "debug: [pred] Before resizing based on leng1. Current points: " << eo_mkpts.size() << ", Target points: " << valid_points_count << std::endl;
        if (valid_points_count > 0 && valid_points_count <= eo_mkpts.size()) {
            eo_mkpts.resize(valid_points_count);
            ir_mkpts.resize(valid_points_count);
        } else if (valid_points_count > eo_mkpts.size()) {
             std::cout << "debug: WARNING: leng1 (" << valid_points_count << ") is greater than the number of keypoints detected (" << eo_mkpts.size() << "). Using all detected points." << std::endl;
        } else {
            std::cout << "debug: WARNING: No valid points to select (leng1=" << valid_points_count << "). Clearing results." << std::endl;
            eo_mkpts.clear();
            ir_mkpts.clear();
        }
        std::cout << "debug: [pred] After resizing based on leng1. Final points: " << eo_mkpts.size() << std::endl;

        // 4. Scale points to output resolution, matching the ONNX version's logic
        std::cout << "debug: [pred] Scaling points with bias. Scale W: " << param_.out_width_scale << ", Scale H: " << param_.out_height_scale << ", Bias X: " << param_.bias_x << ", Bias Y: " << param_.bias_y << std::endl;
        for (auto& pt : eo_mkpts) {
            float scaled_x = pt.x * param_.out_width_scale + param_.bias_x;
            float scaled_y = pt.y * param_.out_height_scale + param_.bias_y;
            pt.x = static_cast<int>(scaled_x);
            pt.y = static_cast<int>(scaled_y);
        }
        for (auto& pt : ir_mkpts) {
            float scaled_x = pt.x * param_.out_width_scale + param_.bias_x;
            float scaled_y = pt.y * param_.out_height_scale + param_.bias_y;
            pt.x = static_cast<int>(scaled_x);
            pt.y = static_cast<int>(scaled_y);
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
            int x_eo = static_cast<int>(eo_kps_raw[i * num_coords + 0]);
            int y_eo = static_cast<int>(eo_kps_raw[i * num_coords + 1]);
            eo_kps.emplace_back(x_eo, y_eo);

            int x_ir = static_cast<int>(ir_kps_raw[i * num_coords + 0]);
            int y_ir = static_cast<int>(ir_kps_raw[i * num_coords + 1]);
            ir_kps.emplace_back(x_ir, y_ir);
        }
        
        std::cout << "debug: Raw leng1=" << leng1_raw << ", leng2=" << leng2_raw << std::endl;
        std::cout << "debug: [runInference] After parsing loop. Parsed " << eo_kps.size() << " raw keypoints." << std::endl;

        // Print first 5 keypoints to check their values
        for (int i = 0; i < std::min(99999, (int)eo_kps.size()); ++i) {
            std::cout << "debug: [runInference] Raw KP " << i << ": EO(" << eo_kps[i].x << "," << eo_kps[i].y 
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

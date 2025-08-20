#include "core_image_align_tensorrt.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

using namespace core;
using namespace std;

std::shared_ptr<ImageAlignTensorRT> ImageAlignTensorRT::create_instance(const Param& param) {
    return std::shared_ptr<ImageAlignTensorRT>(new ImageAlignTensorRT(param));
}

ImageAlignTensorRT::ImageAlignTensorRT(const Param& param) : param_(param) {
    if (!loadEngine(param.engine_path)) {
        throw std::runtime_error("Failed to load TensorRT engine: " + param.engine_path);
    }
    std::cout << "✅ [TensorRT] Successfully loaded TensorRT engine from: " << param.engine_path << std::endl;
}

ImageAlignTensorRT::~ImageAlignTensorRT() {
    // TensorRT 8.6+ 使用智能指針，不需要手動 destroy
    if (context_) {
        delete context_;
        context_ = nullptr;
    }
    if (engine_) {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_) {
        delete runtime_;
        runtime_ = nullptr;
    }
}

bool ImageAlignTensorRT::loadEngine(const std::string& engine_path) {
    std::cout << "🔧 [TensorRT] Loading TensorRT engine: " << engine_path << std::endl;
    
    // 檢查模型文件是否存在
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) {
        std::cerr << "❌ [TensorRT] Engine file not found: " << engine_path << std::endl;
        return false;
    }
    
    // 讀取引擎文件
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    std::cout << "📁 [TensorRT] Engine file size: " << (size / 1024.0 / 1024.0) << " MB" << std::endl;
    
    // 創建 TensorRT runtime 和 engine
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) {
        std::cerr << "❌ [TensorRT] Failed to create runtime" << std::endl;
        return false;
    }
    
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!engine_) {
        std::cerr << "❌ [TensorRT] Failed to deserialize engine" << std::endl;
        return false;
    }
    
    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "❌ [TensorRT] Failed to create execution context" << std::endl;
        return false;
    }
    
    // 顯示引擎信息
    int num_bindings = engine_->getNbBindings();
    std::cout << "📊 [TensorRT] Engine loaded successfully!" << std::endl;
    std::cout << "📊 [TensorRT] Number of bindings: " << num_bindings << std::endl;
    
    // 顯示輸入輸出綁定信息
    for (int i = 0; i < num_bindings; i++) {
        const char* name = engine_->getBindingName(i);
        auto dims = engine_->getBindingDimensions(i);
        bool is_input = engine_->bindingIsInput(i);
        
        std::cout << "  " << (is_input ? "📥" : "📤") << " " 
                  << (is_input ? "Input" : "Output") << " " << i 
                  << ": " << name << " | Shape: (";
        
        for (int j = 0; j < dims.nbDims; j++) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    return true;
}

void ImageAlignTensorRT::align(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_pts, std::vector<cv::Point2i>& ir_pts, cv::Mat& H) {
    try {
        std::cout << "🔍 [TensorRT] ===== Starting TensorRT Alignment =====" << std::endl;
        
        // 預測特徵點 (基於 test_trt_inference.py 的成功實作)
        pred(eo, ir, eo_pts, ir_pts);
        
        std::cout << "🎯 [TensorRT] 共有 " << eo_pts.size() << " 個 EO 特徵點, " << ir_pts.size() << " 個 IR 特徵點" << std::endl;
        
        // 計算 homography
        if (eo_pts.size() >= 4 && ir_pts.size() >= 4) {
            // 轉換為 Point2f 進行 homography 計算
            std::vector<cv::Point2f> eo_pts_f, ir_pts_f;
            for (const auto& pt : eo_pts) {
                eo_pts_f.emplace_back(pt.x, pt.y);
            }
            for (const auto& pt : ir_pts) {
                ir_pts_f.emplace_back(pt.x, pt.y);
            }
            
            // 使用 RANSAC 計算 homography
            cv::Mat mask;
            H = cv::findHomography(eo_pts_f, ir_pts_f, cv::RANSAC, 8.0, mask, 800, 0.98);
            
            if (!H.empty() && cv::determinant(H) > 1e-6) {
                std::cout << "✅ [TensorRT] Successfully computed homography matrix" << std::endl;
                
                // 根據 homography 過濾特徵點
                std::vector<cv::Point2i> filtered_eo_pts, filtered_ir_pts;
                for (size_t i = 0; i < eo_pts.size() && i < mask.rows; i++) {
                    if (mask.at<uchar>(i, 0) > 0) {
                        filtered_eo_pts.push_back(eo_pts[i]);
                        filtered_ir_pts.push_back(ir_pts[i]);
                    }
                }
                
                std::cout << "🔍 [TensorRT] RANSAC 後有效特徵點對: " << filtered_eo_pts.size() << std::endl;
                
                eo_pts = filtered_eo_pts;
                ir_pts = filtered_ir_pts;
            } else {
                std::cout << "❌ [TensorRT] Failed to compute valid homography" << std::endl;
                H = cv::Mat::eye(3, 3, CV_64F);
            }
        } else {
            std::cout << "⚠️ [TensorRT] Insufficient points for homography calculation (" << eo_pts.size() << " points)" << std::endl;
            H = cv::Mat::eye(3, 3, CV_64F);
        }
        
        std::cout << "✅ [TensorRT] ===== TensorRT Alignment Complete =====" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ [TensorRT] Error in alignment: " << e.what() << std::endl;
        H = cv::Mat::eye(3, 3, CV_64F);
    }
}

void ImageAlignTensorRT::pred(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_pts, std::vector<cv::Point2i>& ir_pts) {
    std::cout << "🔍 [TensorRT] Starting inference..." << std::endl;
    
    if (eo.channels() != 1 || ir.channels() != 1) {
        throw std::runtime_error("[TensorRT] eo and ir must be single channel images");
    }
    
    // Resize input images to pred_width x pred_height
    cv::Mat eo_temp, ir_temp;
    cv::resize(eo, eo_temp, cv::Size(param_.input_w, param_.input_h));
    cv::resize(ir, ir_temp, cv::Size(param_.input_w, param_.input_h));
    
    std::cout << "📊 [TensorRT] Input images resized to: " << param_.input_w << "x" << param_.input_h << std::endl;
    
    // 正規化到 [0,1]
    eo_temp.convertTo(eo_temp, CV_32F, 1.0f / 255.0f);
    ir_temp.convertTo(ir_temp, CV_32F, 1.0f / 255.0f);
    
    // 確保記憶體連續性
    eo_temp = eo_temp.clone();
    ir_temp = ir_temp.clone();
    
    std::cout << "📊 [TensorRT] Images normalized, range: [0, 1]" << std::endl;
    std::cout << "📊 [TensorRT] EO image range: [" << eo_temp.ptr<float>()[0] << ", " 
              << *std::max_element(eo_temp.ptr<float>(), eo_temp.ptr<float>() + eo_temp.total()) << "]" << std::endl;
    
    // 清空輸出
    eo_pts.clear();
    ir_pts.clear();
    
    // 運行 TensorRT 推論
    if (!runInference(eo_temp, ir_temp, eo_pts, ir_pts)) {
        std::cerr << "❌ [TensorRT] Inference failed!" << std::endl;
        // 不使用 fallback，讓錯誤明確顯示
        return;
    }
    
    std::cout << "✅ [TensorRT] Inference completed successfully!" << std::endl;
}

bool ImageAlignTensorRT::runInference(const cv::Mat& eo_img, const cv::Mat& ir_img, 
                                      std::vector<cv::Point2i>& eo_pts, std::vector<cv::Point2i>& ir_pts) {
    
    std::cout << "⚡ [TensorRT] Executing inference..." << std::endl;
    
    // 獲取綁定信息
    int num_bindings = engine_->getNbBindings();
    std::vector<void*> buffers(num_bindings);
    
    // 設置輸入
    int input_idx_vi = 0; // 假設第一個輸入是可見光
    int input_idx_ir = 1; // 假設第二個輸入是紅外線
    
    // 為輸入分配 GPU 記憶體並複製數據
    size_t input_size = eo_img.total() * sizeof(float);
    
    void* d_input_vi;
    void* d_input_ir;
    cudaMalloc(&d_input_vi, input_size);
    cudaMalloc(&d_input_ir, input_size);
    
    // 複製數據到 GPU，注意 OpenCV Mat 可能不是連續的
    cv::Mat eo_continuous = eo_img.isContinuous() ? eo_img : eo_img.clone();
    cv::Mat ir_continuous = ir_img.isContinuous() ? ir_img : ir_img.clone();
    
    cudaMemcpy(d_input_vi, eo_continuous.ptr<float>(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_ir, ir_continuous.ptr<float>(), input_size, cudaMemcpyHostToDevice);
    
    buffers[input_idx_vi] = d_input_vi;
    buffers[input_idx_ir] = d_input_ir;
    
    // 為輸出分配記憶體
    std::vector<void*> d_outputs;
    std::vector<size_t> output_sizes;
    
    for (int i = 0; i < num_bindings; i++) {
        if (!engine_->bindingIsInput(i)) {
            auto dims = engine_->getBindingDimensions(i);
            
            size_t output_size = 1;
            for (int j = 0; j < dims.nbDims; j++) {
                output_size *= dims.d[j];
            }
            output_size *= sizeof(float); // 假設輸出是 float
            
            void* d_output;
            cudaMalloc(&d_output, output_size);
            d_outputs.push_back(d_output);
            output_sizes.push_back(output_size);
            buffers[i] = d_output;
            
            std::cout << "📤 [TensorRT] Output " << i << " allocated: " 
                      << (output_size / sizeof(float)) << " elements" << std::endl;
        }
    }
    
    // 執行推論
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    bool success = context_->executeV2(buffers.data());
    
    if (!success) {
        std::cerr << "❌ [TensorRT] Engine execution failed" << std::endl;
        cudaStreamDestroy(stream);
        cudaFree(d_input_vi);
        cudaFree(d_input_ir);
        for (void* d_output : d_outputs) {
            cudaFree(d_output);
        }
        return false;
    }
    
    std::cout << "⚡ [TensorRT] Engine execution successful!" << std::endl;
    
    // 複製輸出回 CPU 並解析特徵點
    // 假設輸出順序: leng1, mkpt0, leng2, mkpt1 (基於 Python 版本)
    if (d_outputs.size() >= 4) {
        // 複製 leng1 (有效 EO 特徵點數量)
        int leng1;
        cudaMemcpy(&leng1, d_outputs[0], sizeof(int), cudaMemcpyDeviceToHost);
        
        // 複製 mkpt0 (EO 特徵點，假設最大 1200 個)
        std::vector<float> mkpt0_data(1200 * 2); // 1200 points * (x,y)
        cudaMemcpy(mkpt0_data.data(), d_outputs[1], output_sizes[1], cudaMemcpyDeviceToHost);
        
        // 複製 leng2 (有效 IR 特徵點數量)
        int leng2;
        cudaMemcpy(&leng2, d_outputs[2], sizeof(int), cudaMemcpyDeviceToHost);
        
        // 複製 mkpt1 (IR 特徵點)
        std::vector<float> mkpt1_data(1200 * 2); // 1200 points * (x,y)
        cudaMemcpy(mkpt1_data.data(), d_outputs[3], output_sizes[3], cudaMemcpyDeviceToHost);
        
        std::cout << "🎯 [TensorRT] leng1 (有效 EO 特徵點): " << leng1 << std::endl;
        std::cout << "🎯 [TensorRT] leng2 (有效 IR 特徵點): " << leng2 << std::endl;
        
        // 提取有效特徵點 (只取前 leng1/leng2 個，過濾 (0,0) 點)
        for (int i = 0; i < std::min(leng1, 1200); i++) {
            float x = mkpt0_data[i * 2];
            float y = mkpt0_data[i * 2 + 1];
            if (x != 0.0f || y != 0.0f) { // 過濾 (0,0) 填充點
                eo_pts.push_back(cv::Point2i(static_cast<int>(x), static_cast<int>(y)));
            }
        }
        
        for (int i = 0; i < std::min(leng2, 1200); i++) {
            float x = mkpt1_data[i * 2];
            float y = mkpt1_data[i * 2 + 1];
            if (x != 0.0f || y != 0.0f) { // 過濾 (0,0) 填充點
                ir_pts.push_back(cv::Point2i(static_cast<int>(x), static_cast<int>(y)));
            }
        }
        
        std::cout << "✅ [TensorRT] 提取到 " << eo_pts.size() << " 個有效 EO 特徵點" << std::endl;
        std::cout << "✅ [TensorRT] 提取到 " << ir_pts.size() << " 個有效 IR 特徵點" << std::endl;
        
        // 顯示特徵點範圍
        if (!eo_pts.empty()) {
            int min_x = std::min_element(eo_pts.begin(), eo_pts.end(), 
                                       [](const cv::Point2i& a, const cv::Point2i& b) { return a.x < b.x; })->x;
            int max_x = std::max_element(eo_pts.begin(), eo_pts.end(), 
                                       [](const cv::Point2i& a, const cv::Point2i& b) { return a.x < b.x; })->x;
            int min_y = std::min_element(eo_pts.begin(), eo_pts.end(), 
                                       [](const cv::Point2i& a, const cv::Point2i& b) { return a.y < b.y; })->y;
            int max_y = std::max_element(eo_pts.begin(), eo_pts.end(), 
                                       [](const cv::Point2i& a, const cv::Point2i& b) { return a.y < b.y; })->y;
            
            std::cout << "📊 [TensorRT] EO 特徵點範圍: x[" << min_x << ", " << max_x << "], y[" << min_y << ", " << max_y << "]" << std::endl;
        }
        
    } else {
        std::cerr << "❌ [TensorRT] Expected 4 outputs, got " << d_outputs.size() << std::endl;
    }
    
    // 清理記憶體
    cudaStreamDestroy(stream);
    cudaFree(d_input_vi);
    cudaFree(d_input_ir);
    for (void* d_output : d_outputs) {
        cudaFree(d_output);
    }
    
    return true;
}

#include "../include/core_image_align_onnxrt.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace core {

class ImageAlignONNXRTImpl : public ImageAlignONNXRT {
private:
    Param param_;
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

public:
    ImageAlignONNXRTImpl(const Param& param) : param_(param), env_(ORT_LOGGING_LEVEL_WARNING, "ImageAlign") {
        // 檢查模型文件是否存在
        if (!std::experimental::filesystem::exists(param_.model_path)) {
            std::cerr << "FATAL ERROR: Model file not found: " << param_.model_path << std::endl;
            throw std::runtime_error("ONNX model file not found: " + param_.model_path);
        }
        
        try {
            // 設定會話選項
            session_options_.SetIntraOpNumThreads(1);
            if (param_.device == "cuda") {
                // ONNX Runtime 1.8.1 API for CUDA
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                session_options_.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "ONNX Runtime loaded with CUDA support" << std::endl;
            } else {
                std::cout << "ONNX Runtime loaded with CPU support" << std::endl;
            }
            
            // 創建會話
            session_ = std::make_unique<Ort::Session>(env_, param_.model_path.c_str(), session_options_);
            
            // 獲取輸入輸出信息
            size_t num_inputs = session_->GetInputCount();
            size_t num_outputs = session_->GetOutputCount();
            
            std::cout << "Model inputs: " << num_inputs << ", outputs: " << num_outputs << std::endl;
            
            // 獲取輸入名稱和形狀
            input_names_.reserve(num_inputs);
            input_shapes_.reserve(num_inputs);
            
            for (size_t i = 0; i < num_inputs; i++) {
                Ort::AllocatorWithDefaultOptions allocator;
                char* input_name = session_->GetInputName(i, allocator);
                input_names_.push_back(input_name);
                
                auto input_type_info = session_->GetInputTypeInfo(i);
                auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                auto input_shape = input_tensor_info.GetShape();
                input_shapes_.push_back(input_shape);
                
                std::cout << "Input " << i << ": " << input_names_[i] << " [";
                for (size_t j = 0; j < input_shape.size(); j++) {
                    std::cout << input_shape[j];
                    if (j < input_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            // 獲取輸出名稱和形狀
            output_names_.reserve(num_outputs);
            output_shapes_.reserve(num_outputs);
            
            for (size_t i = 0; i < num_outputs; i++) {
                Ort::AllocatorWithDefaultOptions allocator;
                char* output_name = session_->GetOutputName(i, allocator);
                output_names_.push_back(output_name);
                
                auto output_type_info = session_->GetOutputTypeInfo(i);
                auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
                auto output_shape = output_tensor_info.GetShape();
                output_shapes_.push_back(output_shape);
                
                std::cout << "Output " << i << ": " << output_names_[i] << " [";
                for (size_t j = 0; j < output_shape.size(); j++) {
                    std::cout << output_shape[j];
                    if (j < output_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            std::cout << "Successfully loaded ONNX model from: " << param_.model_path << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "FATAL ERROR: Failed to load ONNX model: " << e.what() << std::endl;
            throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
        }
    }

    ~ImageAlignONNXRTImpl() {
        // 釋放分配的名稱 (ONNX Runtime 1.8.1 uses regular char*)
        for (auto& name : input_names_) {
            free((void*)name);
        }
        for (auto& name : output_names_) {
            free((void*)name);
        }
    }

    bool align(const cv::Mat& eo, const cv::Mat& ir,
              std::vector<cv::Point2i>& eo_pts,
              std::vector<cv::Point2i>& ir_pts,
              cv::Mat& H) override {
        
        eo_pts.clear();
        ir_pts.clear();

        try {
            // 準備輸入數據
            cv::Mat eo_resized, ir_resized;
            cv::resize(eo, eo_resized, cv::Size(param_.pred_width, param_.pred_height));
            cv::resize(ir, ir_resized, cv::Size(param_.pred_width, param_.pred_height));

            // 轉換為灰階（如果需要）
            if (eo_resized.channels() == 3) {
                cv::cvtColor(eo_resized, eo_resized, cv::COLOR_BGR2GRAY);
            }
            if (ir_resized.channels() == 3) {
                cv::cvtColor(ir_resized, ir_resized, cv::COLOR_BGR2GRAY);
            }

            // 歸一化為 [0, 1]
            eo_resized.convertTo(eo_resized, CV_32F, 1.0/255.0);
            ir_resized.convertTo(ir_resized, CV_32F, 1.0/255.0);

            // 準備輸入張量
            std::vector<float> vi_data, ir_data;
            
            // 將 cv::Mat 轉換為平鋪的數組 (NCHW 格式)
            size_t input_size = param_.pred_height * param_.pred_width;
            vi_data.resize(input_size);
            ir_data.resize(input_size);
            
            for (int i = 0; i < param_.pred_height; i++) {
                for (int j = 0; j < param_.pred_width; j++) {
                    vi_data[i * param_.pred_width + j] = eo_resized.at<float>(i, j);
                    ir_data[i * param_.pred_width + j] = ir_resized.at<float>(i, j);
                }
            }

            // 創建輸入張量
            std::vector<int64_t> input_shape = {1, 1, param_.pred_height, param_.pred_width};
            
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, vi_data.data(), vi_data.size(), input_shape.data(), input_shape.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info, ir_data.data(), ir_data.size(), input_shape.data(), input_shape.size()));

            // 執行推理
            auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, 
                                              input_names_.data(), input_tensors.data(), input_tensors.size(),
                                              output_names_.data(), output_names_.size());

            // 檢查輸出
            if (output_tensors.size() < 6) {
                std::cerr << "Expected 6 outputs but got " << output_tensors.size() << std::endl;
                return false;
            }

            // 獲取關鍵點
            auto mkpts0_tensor = &output_tensors[0];
            auto mkpts1_tensor = &output_tensors[1];
            auto score_tensor = &output_tensors[5];

            // 獲取數據指針
            const float* mkpts0_data = mkpts0_tensor->GetTensorData<float>();
            const float* mkpts1_data = mkpts1_tensor->GetTensorData<float>();
            const float* score_data = score_tensor->GetTensorData<float>();

            // 獲取張量形狀
            auto mkpts0_shape = mkpts0_tensor->GetTensorTypeAndShapeInfo().GetShape();
            auto mkpts1_shape = mkpts1_tensor->GetTensorTypeAndShapeInfo().GetShape();
            auto score_shape = score_tensor->GetTensorTypeAndShapeInfo().GetShape();

            std::cout << "Raw output shapes:" << std::endl;
            std::cout << "  mkpts0: [" << mkpts0_shape[0] << ", " << mkpts0_shape[1] << "]" << std::endl;
            std::cout << "  mkpts1: [" << mkpts1_shape[0] << ", " << mkpts1_shape[1] << "]" << std::endl;
            std::cout << "  score: [" << score_shape[0] << "]" << std::endl;

            int num_points = mkpts0_shape[0];  // 實際的點數量
            
            std::cout << "Detected " << num_points << " keypoint pairs" << std::endl;

            if (num_points == 0) {
                std::cout << "No keypoints detected" << std::endl;
                return false;
            }

            // 轉換特徵點並應用縮放 - 添加詳細調試信息
            for (int i = 0; i < num_points; i++) {
                // 檢查分數是否有效
                float score = score_data[i];
                
                // 獲取原始座標
                float x1_raw = mkpts0_data[i * 2 + 0];
                float y1_raw = mkpts0_data[i * 2 + 1];
                float x2_raw = mkpts1_data[i * 2 + 0];
                float y2_raw = mkpts1_data[i * 2 + 1];
                
                // 應用縮放
                float x1 = x1_raw * param_.out_width_scale + param_.bias_x;
                float y1 = y1_raw * param_.out_height_scale + param_.bias_y;
                float x2 = x2_raw * param_.out_width_scale + param_.bias_x;
                float y2 = y2_raw * param_.out_height_scale + param_.bias_y;
                
                // 調試輸出前幾個點
                if (i < 5) {
                    std::cout << "Point " << i << ": raw(" << x1_raw << "," << y1_raw 
                              << ") -> scaled(" << x1 << "," << y1 << "), score=" << score << std::endl;
                }
                
                // 檢查分數閾值（降低閾值以獲得更多點）
                if (score < 0.01) continue;
                
                // 檢查點的有效性
                float max_w = param_.pred_width * param_.out_width_scale;
                float max_h = param_.pred_height * param_.out_height_scale;
                
                if (x1 >= 0 && x1 < max_w && y1 >= 0 && y1 < max_h &&
                    x2 >= 0 && x2 < max_w && y2 >= 0 && y2 < max_h) {
                    
                    eo_pts.emplace_back((int)x1, (int)y1);
                    ir_pts.emplace_back((int)x2, (int)y2);
                } else {
                    if (i < 5) {
                        std::cout << "Point " << i << " rejected: out of bounds" << std::endl;
                    }
                }
            }

            std::cout << "Valid keypoints after filtering: " << eo_pts.size() << std::endl;

            // ===== 新增視覺化功能：畫出特徵點和匹配線 =====
            if (eo_pts.size() > 0) {
                cv::Mat vis_img;
                drawKeyPointMatches(eo, ir, eo_pts, ir_pts, vis_img);
                
                // 保存視覺化結果
                static int frame_counter = 0;
                std::string vis_path = "/circ330/forgithub/VisualFusion_libtorch/Onnx/output/keypoints_" + 
                                      std::to_string(frame_counter++) + ".jpg";
                cv::imwrite(vis_path, vis_img);
                std::cout << "Saved keypoint visualization to: " << vis_path << std::endl;
            }
            // ===== 視覺化功能結束 =====

            // 過濾特徵點
            filterPoints(eo_pts, ir_pts);

            // 計算Homography矩陣
            if (eo_pts.size() >= 4 && ir_pts.size() >= 4) {
                std::vector<cv::Point2f> eo_pts_f, ir_pts_f;
                for (const auto& pt : eo_pts) eo_pts_f.emplace_back(pt.x, pt.y);
                for (const auto& pt : ir_pts) ir_pts_f.emplace_back(pt.x, pt.y);
                
                cv::Mat mask;
                H = cv::findHomography(eo_pts_f, ir_pts_f, cv::RANSAC, 3.0, mask);
                
                if (!H.empty() && cv::determinant(H) > 1e-6) {
                    std::cout << "Successfully computed homography matrix" << std::endl;
                    return true;
                }
            }

            std::cout << "Insufficient keypoints for homography computation" << std::endl;

        } catch (const Ort::Exception& e) {
            std::cerr << "Error in ONNX Runtime alignment: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in ONNX Runtime alignment: " << e.what() << std::endl;
        }

        return false;
    }

private:
    void drawKeyPointMatches(const cv::Mat& img1, const cv::Mat& img2, 
                           const std::vector<cv::Point2i>& pts1, 
                           const std::vector<cv::Point2i>& pts2, 
                           cv::Mat& output) {
        
        // 創建並排顯示的圖像
        cv::Mat img1_color, img2_color;
        if (img1.channels() == 1) {
            cv::cvtColor(img1, img1_color, cv::COLOR_GRAY2BGR);
        } else {
            img1_color = img1.clone();
        }
        
        if (img2.channels() == 1) {
            cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
        } else {
            img2_color = img2.clone();
        }
        
        // 確保兩個圖像大小相同
        if (img1_color.size() != img2_color.size()) {
            cv::resize(img2_color, img2_color, img1_color.size());
        }
        
        // 創建並排圖像
        cv::Mat combined;
        cv::hconcat(img1_color, img2_color, combined);
        
        // 畫特徵點
        int offset_x = img1_color.cols;  // 第二個圖像的X偏移
        
        for (size_t i = 0; i < pts1.size() && i < pts2.size(); i++) {
            // 生成隨機顏色
            cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
            
            // 畫第一個圖像的特徵點
            cv::circle(combined, pts1[i], 5, color, 2);
            cv::putText(combined, std::to_string(i), 
                       cv::Point(pts1[i].x + 8, pts1[i].y), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            
            // 畫第二個圖像的特徵點（加上偏移）
            cv::Point2i pt2_offset(pts2[i].x + offset_x, pts2[i].y);
            cv::circle(combined, pt2_offset, 5, color, 2);
            cv::putText(combined, std::to_string(i), 
                       cv::Point(pt2_offset.x + 8, pt2_offset.y), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            
            // 畫連接線
            cv::line(combined, pts1[i], pt2_offset, color, 2);
        }
        
        // 添加標題
        cv::putText(combined, "VI Image", cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(combined, "IR Image", cv::Point(offset_x + 10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        
        // 添加統計信息
        std::string info = "Total matches: " + std::to_string(pts1.size());
        cv::putText(combined, info, cv::Point(10, combined.rows - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        output = combined;
    }

    void filterPoints(std::vector<cv::Point2i>& eo_pts, 
                     std::vector<cv::Point2i>& ir_pts) {
        if (eo_pts.size() != ir_pts.size() || eo_pts.empty()) {
            return;
        }

        std::vector<cv::Point2i> filtered_eo_pts, filtered_ir_pts;
        std::vector<float> angles, distances;

        // 計算角度和距離
        for (size_t i = 0; i < eo_pts.size(); i++) {
            float dx = ir_pts[i].x - eo_pts[i].x;
            float dy = ir_pts[i].y - eo_pts[i].y;
            float distance = std::sqrt(dx*dx + dy*dy);
            float angle = std::atan2(dy, dx);
            
            if (distance < param_.distance_mean) {
                angles.push_back(angle);
                distances.push_back(distance);
                filtered_eo_pts.push_back(eo_pts[i]);
                filtered_ir_pts.push_back(ir_pts[i]);
            }
        }

        // 根據角度一致性過濾
        if (!angles.empty()) {
            // 計算角度中位數
            std::vector<float> sorted_angles = angles;
            std::sort(sorted_angles.begin(), sorted_angles.end());
            float median_angle = sorted_angles[sorted_angles.size()/2];

            std::vector<cv::Point2i> temp_eo_pts, temp_ir_pts;
            for (size_t i = 0; i < filtered_eo_pts.size(); i++) {
                float angle_diff = std::abs(angles[i] - median_angle);
                if (angle_diff > M_PI) angle_diff = 2*M_PI - angle_diff;
                
                if (angle_diff <= param_.angle_mean) {
                    temp_eo_pts.push_back(filtered_eo_pts[i]);
                    temp_ir_pts.push_back(filtered_ir_pts[i]);
                }
            }

            eo_pts = temp_eo_pts;
            ir_pts = temp_ir_pts;
        } else {
            eo_pts.clear();
            ir_pts.clear();
        }
    }
};

ImageAlignONNXRT::ptr ImageAlignONNXRT::create_instance(const Param& param) {
    return std::make_shared<ImageAlignONNXRTImpl>(param);
}

}

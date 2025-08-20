#include "../include/core_image_align_onnx.h"
#include <algorithm>
#include <cmath>
#include <iostream>
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

public:
    ImageAlignONNXImpl(const Param& param) : param_(param), env_(ORT_LOGGING_LEVEL_WARNING, "ImageAlign") {
        // 檢查模型文件是否存在
        if (!std::experimental::filesystem::exists(param_.model_path)) {
            std::cerr << "FATAL ERROR: Model file not found: " << param_.model_path << std::endl;
            throw std::runtime_error("ONNX model file not found: " + param_.model_path);
        }
        
        try {
            // 創建 ONNX Runtime session
            session_options_.SetIntraOpNumThreads(1);
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            session_ = std::make_unique<Ort::Session>(env_, param_.model_path.c_str(), session_options_);
            memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
            
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

    // predict keypoints using cpu
    void pred_cpu(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts) {
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

        // run the model
        auto pred = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), inputs.data(), 2, output_names_.data(), output_names_.size());

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

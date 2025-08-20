#include "core_image_align_onnx_polar.h"
#include <algorithm>
#include <cmath>

namespace core {

class ImageAlignONNXImpl : public ImageAlignONNX {
public:
    ImageAlignONNXImpl(const Param& param) : param_(param) {
        try {
            net_ = cv::dnn::readNetFromONNX(param_.model_path);
            
            if (param_.device == "cuda") {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            } else {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
            
            std::cout << "Successfully loaded ONNX model from: " << param_.model_path << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Failed to load ONNX model: " << e.what() << std::endl;
            throw;
        }
    }

    bool align(const cv::Mat& eo, const cv::Mat& ir,
              std::vector<cv::Point2i>& eo_pts,
              std::vector<cv::Point2i>& ir_pts,
              cv::Mat& H) override {
        
        // Prepare input
        cv::Mat blob;
        std::vector<cv::Mat> images = {eo, ir};
        cv::dnn::blobFromImages(images, blob, 1.0/255.0, 
                               cv::Size(param_.pred_width, param_.pred_height), 
                               cv::Scalar(0,0,0), true, false);

        // Forward pass
        net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());

        // Process output
        if (outputs.empty()) {
            std::cout << "No output from the network" << std::endl;
            return false;
        }

        // Assuming the output is a matrix of matched points
        cv::Mat output = outputs[0].reshape(1, outputs[0].total() / 4);
        
        // Extract points
        eo_pts.clear();
        ir_pts.clear();
        
        for (int i = 0; i < output.rows; i++) {
            float x1 = output.at<float>(i, 0) * param_.output_width;
            float y1 = output.at<float>(i, 1) * param_.output_height;
            float x2 = output.at<float>(i, 2) * param_.output_width;
            float y2 = output.at<float>(i, 3) * param_.output_height;
            
            // Add bias
            x1 += param_.bias_x;
            x2 += param_.bias_x;
            y1 += param_.bias_y;
            y2 += param_.bias_y;
            
            eo_pts.emplace_back(x1, y1);
            ir_pts.emplace_back(x2, y2);
        }

        // Filter points based on distance and angle criteria
        filterPoints(eo_pts, ir_pts);

        // Calculate homography if we have enough points
        if (eo_pts.size() >= 4) {
            std::vector<cv::Point2f> eo_pts_f(eo_pts.begin(), eo_pts.end());
            std::vector<cv::Point2f> ir_pts_f(ir_pts.begin(), ir_pts.end());
            
            H = cv::findHomography(eo_pts_f, ir_pts_f, cv::RANSAC);
            return !H.empty();
        }

        return false;
    }

private:
    void filterPoints(std::vector<cv::Point2i>& eo_pts, 
                     std::vector<cv::Point2i>& ir_pts) {
        std::vector<cv::Point2i> filtered_eo_pts, filtered_ir_pts;
        std::vector<float> angles, distances;

        // Calculate angles and distances
        for (size_t i = 0; i < eo_pts.size(); i++) {
            float dx = ir_pts[i].x - eo_pts[i].x;
            float dy = ir_pts[i].y - eo_pts[i].y;
            float distance = std::sqrt(dx*dx + dy*dy);
            float angle = std::atan2(dy, dx) * 180.0f / CV_PI;
            
            if (distance < param_.distance_max) {
                angles.push_back(angle);
                distances.push_back(distance);
                filtered_eo_pts.push_back(eo_pts[i]);
                filtered_ir_pts.push_back(ir_pts[i]);
            }
        }

        // Sort angles and find median
        if (!angles.empty()) {
            std::sort(angles.begin(), angles.end());
            float median_angle = angles[angles.size()/2];

            // Filter based on angle difference from median
            std::vector<cv::Point2i> temp_eo_pts, temp_ir_pts;
            for (size_t i = 0; i < filtered_eo_pts.size(); i++) {
                if (std::abs(angles[i] - median_angle) <= param_.angle_mean) {
                    temp_eo_pts.push_back(filtered_eo_pts[i]);
                    temp_ir_pts.push_back(filtered_ir_pts[i]);
                }
            }

            eo_pts = temp_eo_pts;
            ir_pts = temp_ir_pts;
        }
    }

    Param param_;
    cv::dnn::Net net_;
};

std::shared_ptr<ImageAlignONNX> ImageAlignONNX::create_instance(const Param& param) {
    return std::make_shared<ImageAlignONNXImpl>(param);
}

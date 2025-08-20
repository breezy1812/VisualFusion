#pragma once
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace core {
class ImageAlignTensorRT {
public:
    struct Param {
        int input_w = 320;
        int input_h = 240;
        int output_w = 320;
        int output_h = 240;
        float out_width_scale = 1.0f;
        float out_height_scale = 1.0f;
        float bias_x = 0.0f;
        float bias_y = 0.0f;
        std::string engine_path;
        Param& set_size(int iw, int ih, int ow, int oh) {
            input_w = iw; input_h = ih; output_w = ow; output_h = oh;
            return *this;
        }
        Param& set_scale_and_bias(float scale_w, float scale_h, float bx, float by) {
            out_width_scale = scale_w;
            out_height_scale = scale_h;
            bias_x = bx;
            bias_y = by;
            return *this;
        }
        Param& set_engine(const std::string& path) { engine_path = path; return *this; }
    };
    static std::shared_ptr<ImageAlignTensorRT> create_instance(const Param& param);
    
    // The main alignment function. It takes two images, finds matching keypoints,
    // and computes the homography matrix.
    // The keypoint vectors (eo_pts, ir_pts) are cleared and then filled with the results.
    virtual void align(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_pts, std::vector<cv::Point2i>& ir_pts, cv::Mat& H) = 0;
    
    virtual ~ImageAlignTensorRT();

protected: // Changed from private to protected to allow implementation class to inherit
    ImageAlignTensorRT(const Param& param);

private:
    // PIMPL (Pointer to implementation) pattern can be used here if we want to hide all private members
    // For now, we keep it simple. The implementation will be in the .cpp file.
};
} // namespace core

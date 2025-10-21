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

        int pred_width = 320;
        int pred_height = 240;
        int output_w = 320;
        int output_h = 240;
        float out_width_scale = 1.0f;
        float out_height_scale = 1.0f;
        float bias_x = 0.0f;
        float bias_y = 0.0f;
        std::string engine_path;
        std::string pred_mode = "fp32";
        std::string image_name = "";

        Param& set_size(int pw, int ph, int ow, int oh) {
            pred_width = pw; pred_height = ph; output_w = ow; output_h = oh;

            out_width_scale = ow / (float)pw;
            out_height_scale = oh / (float)ph;
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
        Param& set_pred_mode(const std::string& mode) { pred_mode = mode; return *this; }
        Param& set_image_name(const std::string& name) { image_name = name; return *this; }
    };
    static std::shared_ptr<ImageAlignTensorRT> create_instance(const Param& param);

    virtual void align(const cv::Mat& eo, const cv::Mat& ir, std::vector<cv::Point2i>& eo_pts, std::vector<cv::Point2i>& ir_pts, cv::Mat& H) = 0;

    virtual void set_current_image_name(const std::string& image_name) = 0;
    virtual ~ImageAlignTensorRT();

protected:
    ImageAlignTensorRT(const Param& param);

private:

};
}

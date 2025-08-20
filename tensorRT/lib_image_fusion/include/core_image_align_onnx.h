#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <memory>
#include <experimental/filesystem>

namespace core {

class ImageAlignONNX {
public:
    using ptr = std::shared_ptr<ImageAlignONNX>;

#define degree_to_rad(degree) ((degree) * M_PI / 180.0);

    struct Param {
        // 預測尺寸
        int pred_width = 0;
        int pred_height = 0;

        // 小視窗尺寸
        int small_window_width = 0;
        int small_window_height = 0;

        // 切割視窗尺寸
        int clip_window_width = 0;
        int clip_window_height = 0;

        // 輸出座標縮放尺寸
        float out_width_scale = 1.0;
        float out_height_scale = 1.0;

        int bias_x = 0;
        int bias_y = 0;

        // 模型
        std::string device = "cpu";
        std::string model_path = "";

        // 前一幀距離、水平/垂直篩選距離、平均角度範圍、排序角度範圍
        float distance_last = 10.0;
        float distance_line = 10.0;
        float distance_mean = 20.0;
        float angle_mean = degree_to_rad(10.0);
        float angle_sort = 0.6;

        Param &set_size(int pw, int ph, int ow, int oh) {
            pred_width = pw;
            pred_height = ph;

            small_window_width = pw / 8;
            small_window_height = ph / 8;

            out_width_scale = ow / (float)pw;
            out_height_scale = oh / (float)ph;

            clip_window_width = small_window_width / 10;
            clip_window_height = small_window_height / 10;
            return *this;
        }

        Param &set_model(std::string device, std::string model_path) {
            // 檢查模型
            if (!std::experimental::filesystem::exists(model_path))
                throw std::invalid_argument("Model file not found");
            else
                this->model_path = model_path;

            // 設定裝置
            if (device.compare("cpu") == 0 || device.compare("cuda") == 0)
                this->device = device;
            else
                throw std::invalid_argument("Device not supported");

            return *this;
        }

        Param &set_distance(float line, float last, float mean) {
            distance_line = line;
            distance_last = last;
            distance_mean = mean;
            return *this;
        }

        Param &set_angle(float mean, float sort) {
            angle_mean = degree_to_rad(mean);
            angle_sort = sort;
            return *this;
        }

        Param &set_bias(int x, int y) {
            bias_x = x;
            bias_y = y;
            return *this;
        }
    };

    static ptr create_instance(const Param &param);

    virtual bool align(const cv::Mat &eo, const cv::Mat &ir,
                      std::vector<cv::Point2i> &eo_pts,
                      std::vector<cv::Point2i> &ir_pts,
                      cv::Mat &H) = 0;

    virtual ~ImageAlignONNX() = default;
};

}

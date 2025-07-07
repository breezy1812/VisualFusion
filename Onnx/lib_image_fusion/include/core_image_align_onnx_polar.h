#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <memory>

namespace core {
    class ImageAlignONNX {
    public:
        class Param {
        public:
            Param& set_size(int pred_w, int pred_h, int out_w, int out_h) {
                pred_width = pred_w;
                pred_height = pred_h;
                output_width = out_w;
                output_height = out_h;
                return *this;
            }

            Param& set_model(const std::string& device_, const std::string& model_path_) {
                device = device_;
                model_path = model_path_;
                return *this;
            }

            Param& set_bias(int x, int y) {
                bias_x = x;
                bias_y = y;
                return *this;
            }

            Param& set_distance(float line_, float last_, float max_) {
                distance_line = line_;
                distance_last = last_;
                distance_max = max_;
                return *this;
            }

            Param& set_angle(float mean_, float sort_) {
                angle_mean = mean_;
                angle_sort = sort_;
                return *this;
            }

            int pred_width = 480;
            int pred_height = 360;
            int output_width = 480;
            int output_height = 360;
            std::string device = "cpu";
            std::string model_path = "";
            int bias_x = 0;
            int bias_y = 0;
            float distance_line = 10.0f;
            float distance_last = 15.0f;
            float distance_max = 20.0f;
            float angle_mean = 10.0f;
            float angle_sort = 0.7f;
        };

        static std::shared_ptr<ImageAlignONNX> create_instance(const Param& param = Param());

        virtual bool align(const cv::Mat& eo, const cv::Mat& ir,
                         std::vector<cv::Point2i>& eo_pts,
                         std::vector<cv::Point2i>& ir_pts,
                         cv::Mat& H) = 0;

        virtual ~ImageAlignONNX() = default;
    };
}

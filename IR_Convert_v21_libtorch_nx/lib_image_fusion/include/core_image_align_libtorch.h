

#ifndef INCLUDE_CORE_IMAGE_ALIGN_LIBTORCH_H_
#define INCLUDE_CORE_IMAGE_ALIGN_LIBTORCH_H_

#include <memory>
#include <string>
#include <experimental/filesystem>

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

namespace core
{

  class ImageAlign
  {
  public:
    using ptr = std::shared_ptr<ImageAlign>;

#define degree_to_rad(degree) ((degree) * M_PI / 180.0);

    struct Param
    {

      int pred_width = 0;
      int pred_height = 0;

      int small_window_width = 0;
      int small_window_height = 0;

      int clip_window_width = 0;
      int clip_window_height = 0;

      float out_width_scale = 1.0;
      float out_height_scale = 1.0;

      int bias_x = 0;
      int bias_y = 0;

      std::string mode = "fp32";
      std::string device = "cpu";
      std::string model_path = "";

      float distance_last = 10.0;
      float distance_line = 10.0;
      float distance_mean = 20.0;
      float angle_mean = degree_to_rad(10.0);
      float angle_sort = 0.6;

      Param &set_size(int pw, int ph, int ow, int oh)
      {
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

      Param &set_net(std::string device, std::string model_path, std::string mode = "fp32")
      {

        if (!std::experimental::filesystem::exists(model_path))
          throw std::invalid_argument("Model file not found");
        else
          this->model_path = model_path;

        if (device.compare("cpu") == 0 || device.compare("cuda") == 0)
          this->device = device;
        else
          throw std::invalid_argument("Device not supported");

        if (mode.compare("fp32") == 0 || mode.compare("fp16") == 0)
          this->mode = mode;
        else
          throw std::invalid_argument("Model output mode not supported");

        return *this;
      }

      Param &set_distance(float line, float last, float mean)
      {
        distance_line = line;
        distance_last = last;
        distance_mean = mean;
        return *this;
      }

      Param &set_angle(float mean, float sort)
      {
        angle_mean = degree_to_rad(mean);
        angle_sort = sort;
        return *this;
      }

      Param &set_bias(int x, int y)
      {
        bias_x = x;
        bias_y = y;
        return *this;
      }
    };

    static ptr create_instance(const Param &param)
    {
      return std::make_shared<ImageAlign>(std::move(param));
    }

    explicit ImageAlign(Param param);

    void align(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, cv::Mat &H, const std::string& filename = "");

  private:
    Param param_;

    torch::Device device{torch::kCPU};

    torch::jit::script::Module net;

    void warm_up();

    void smart_warmup();

    void pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, const std::string& filename = "");

    std::vector<float> line_equation(cv::Point2f &pt1, cv::Point2f &pt2);

    int judge_h_line(std::vector<float> &line, cv::Point2i &pt);
    int judge_v_line(std::vector<float> &line, cv::Point2i &pt);

    int judge_quadrant(std::vector<float> &line_v, std::vector<float> &line_h, cv::Point2i &pt);

    void show(std::vector<std::vector<int>> q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<float> v_line, std::vector<float> h_line);

    void class_quadrant(std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff);

    void combine_quadrant(std::vector<std::vector<int>> &q_idx);

    bool check_quadrant_imbalance(std::vector<std::vector<int>> &q_idx);

    void find_line(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<float> &v_line, std::vector<float> &h_line);

    void apply_keypoints(std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, std::vector<int> &filter_idx);

    void apply_quadrants(std::vector<std::vector<int>> &q_idx, std::vector<int> &filter_idx);

    float distance_line(std::vector<float> &line, cv::Point2i &pt);

    std::vector<int> filter_diagonal(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts);

    std::vector<int> filter_same(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts);

    std::vector<int> filter_distance(std::vector<float> &v_line, std::vector<float> &h_line, std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts);

    std::vector<int> filter_mean_angle(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff);
    std::vector<int> filter_sort_angle(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff);

    std::vector<int> filter_mean_distance(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &q_diff);

    std::vector<int> filter_last_H(std::vector<std::vector<int>> &q_idx, std::vector<cv::Point2i> &eo_mkpts, std::vector<cv::Point2i> &ir_mkpts, cv::Mat &H);

    void writeTimingToCSV(const std::string& operation, double time_ms, int leng, const std::string& filename = "");
  };

} 

#endif 

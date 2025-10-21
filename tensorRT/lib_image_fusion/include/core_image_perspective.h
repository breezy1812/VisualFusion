

#ifndef INCLUDE_CORE_IMAGE_PERSPECTIVE_H_
#define INCLUDE_CORE_IMAGE_PERSPECTIVE_H_

#include <memory>
#include <random>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace core
{

  class ImagePerspective
  {
  public:
    using ptr = std::shared_ptr<ImagePerspective>;

    struct Param
    {
      cv::Mat H;
      cv::Mat wrap(cv::Mat &in, cv::Mat &H, int width, int height);

      bool check = true;
      float distance = 10;
      float accuracy = 0.7;

      int msac_iteration = 1000;
      float msac_threshold = 3.0;

      Param &set_check(bool chk, float acc, float dis)
      {
        check = chk;
        accuracy = acc;
        distance = dis;
        return *this;
      }
    };

    static ptr create_instance(Param param)
    {
      return std::make_shared<ImagePerspective>(param);
    }

    ImagePerspective(Param param);

    cv::Mat wrap(cv::Mat &in, int width, int height);

    bool find_perspective_matrix(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst);
    bool find_perspective_matrix_msac(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst);

    int count_allow(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst, cv::Mat &H);

    float calculate_mse(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst, cv::Mat &H);

    cv::Mat get_perspective_matrix()
    {
      return param_.H;
    }

  private:
    Param param_;
  };

} 

#endif 

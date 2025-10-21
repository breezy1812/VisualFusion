

#include <core_image_to_gray.h>

namespace core
{

  ImageToGray::ImageToGray(Param param) : param_(std::move(param))
  {
  }

  cv::Mat ImageToGray::gray(cv::Mat &in)
  {
    cv::Mat out;
    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
    return out;
  }

} 

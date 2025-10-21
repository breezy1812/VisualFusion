

#include <core_image_perspective.h>

namespace core
{

  ImagePerspective::ImagePerspective(Param param) : param_(std::move(param)) {}

  cv::Mat ImagePerspective::wrap(cv::Mat &in, int width, int height)
  {
    if (param_.H.empty())
      return in;

    cv::Mat out;
    cv::warpPerspective(in, out, param_.H, cv::Size(width, height));
    return out;
  }

  bool ImagePerspective::find_perspective_matrix(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst)
  {

    if (src.size() < 4 || dst.size() < 4)
      return false;

    if (src.size() != dst.size())
      return false;

    cv::Mat H;

    try
    {

      H = cv::findHomography(src, dst, cv::RANSAC);
    }
    catch (const std::exception &e)
    {
      return false;
    }

    if (H.empty())
      return false;

    if (param_.check)
    {
      float score = (float)count_allow(src, dst, H) / src.size();
      if (score < param_.accuracy)
        return false;
    }

    param_.H = H;
    return true;
  }

  bool ImagePerspective::find_perspective_matrix_msac(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst)
  {

    if (src.size() < 4 || dst.size() < 4)
      return false;

    if (src.size() != dst.size())
      return false;

    int counts = 0;
    int max_iteration = param_.msac_iteration;

    cv::Mat best_H;
    float best_cost = std::numeric_limits<float>::max();

    while (counts < max_iteration)
    {

      std::vector<int> idx;
      while (idx.size() < 4)
      {
        int r = rand() % src.size();
        if (std::find(idx.begin(), idx.end(), r) == idx.end())
          idx.push_back(r);
      }

      std::vector<cv::Point2i> src_4, dst_4;
      for (int i = 0; i < 4; i++)
      {
        src_4.emplace_back(src[idx[i]]);
        dst_4.emplace_back(dst[idx[i]]);
      }

      cv::Mat H;
      try
      {
        H = cv::findHomography(src_4, dst_4);
      }
      catch (const std::exception &e)
      {
        continue;
      }

      if (H.empty())
        continue;

      double a = H.at<double>(0, 0), b = H.at<double>(0, 1), c = H.at<double>(0, 2);
      double d = H.at<double>(1, 0), e = H.at<double>(1, 1), f = H.at<double>(1, 2);
      double g = H.at<double>(2, 0), h = H.at<double>(2, 1);

      float cost = 0;
      for (int i = 0; i < src.size(); i++)
      {
        int x = src[i].x, y = src[i].y;
        float z_bar = g * x + h * y + 1;
        float x_bar = (a * x + b * y + c) / z_bar;
        float y_bar = (d * x + e * y + f) / z_bar;

        float dis_x = x_bar - dst[i].x;
        float dis_y = y_bar - dst[i].y;
        float dis = std::sqrt(dis_x * dis_x + dis_y * dis_y);

        if (dis > param_.msac_threshold)
          cost += param_.msac_threshold;
        else
          cost += dis;
      }

      if (cost < best_cost)
      {
        best_cost = cost;
        best_H = H;
      }

      counts++;
    }

    if (best_H.empty())
      return false;

    param_.H = best_H;

    return true;
  }

  int ImagePerspective::count_allow(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst, cv::Mat &H)
  {
    int count = 0;

    double a = H.at<double>(0, 0), b = H.at<double>(0, 1), c = H.at<double>(0, 2);
    double d = H.at<double>(1, 0), e = H.at<double>(1, 1), f = H.at<double>(1, 2);
    double g = H.at<double>(2, 0), h = H.at<double>(2, 1);

    for (int i = 0; i < src.size(); i++)
    {
      int x = src[i].x, y = src[i].y;
      float z_bar = g * x + h * y + 1;
      float x_bar = (a * x + b * y + c) / z_bar;
      float y_bar = (d * x + e * y + f) / z_bar;

      float dis_x = x_bar - dst[i].x;
      float dis_y = y_bar - dst[i].y;
      float dis = sqrt(dis_x * dis_x + dis_y * dis_y);

      if (dis <= param_.distance)
        count++;
    }

    return count;
  }

  float ImagePerspective::calculate_mse(std::vector<cv::Point2i> &src, std::vector<cv::Point2i> &dst, cv::Mat &H)
  {
    float score = 0;

    double a = H.at<double>(0, 0), b = H.at<double>(0, 1), c = H.at<double>(0, 2);
    double d = H.at<double>(1, 0), e = H.at<double>(1, 1), f = H.at<double>(1, 2);
    double g = H.at<double>(2, 0), h = H.at<double>(2, 1);

    for (int i = 0; i < src.size(); i++)
    {
      int x = src[i].x, y = src[i].y;
      float z_bar = g * x + h * y + 1;
      float x_bar = (a * x + b * y + c) / z_bar;
      float y_bar = (d * x + e * y + f) / z_bar;

      float dis_x = x_bar - dst[i].x;
      float dis_y = y_bar - dst[i].y;

      score += sqrt(dis_x * dis_x + dis_y * dis_y);
    }

    return score;
  }
} 

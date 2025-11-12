#include <core_image_fusion.h>

namespace core
{

  ImageFusion::ImageFusion(Param param) : param_(std::move(param)) {}

  cv::Mat ImageFusion::equalization(cv::Mat &in)
  {
    cv::Mat out = in.clone();

    cv::Mat sum;
    int histSize = 256;
    float range[2] = {0, 256};
    const float *hisRange = {range};
    cv::calcHist(&in, 1, 0, cv::Mat(), sum, 1, &histSize, &hisRange);

    cv::Scalar mean = cv::mean(in);

    int th = param_.threshold_equalization;
    int th0 = param_.threshold_equalization_zero;
    int th1 = param_.threshold_equalization_low;
    int th2 = param_.threshold_equalization_high;

    if (mean[0] <= th)
    {
      cv::Mat table(1, 256, CV_8U);
      unsigned char *tb = table.data;

      int min = 0;
      while (sum.at<float>(min) == 0)
        min++;

      min = std::max(min, th0);
      th1 = std::max(th1, min);

      int range = th2 - th1;

      int pn = 0;
      for (int i = th1; i <= th2; i++)
        pn += sum.at<float>(i);

      float prob = 0.0;
      for (int i = 0; i < 256; i++)
      {
        if (i < min)
          tb[i] = 0;
        else if (th1 <= i && i < th2)
        {
          prob += sum.at<float>(i) / pn;
          tb[i] = prob * range + th1;
        }
        else
          tb[i] = i;
      }

      cv::LUT(in, table, out);
    }

    return out;
  }

  cv::Mat ImageFusion::edge(cv::Mat &in)
  {
    cv::Mat out;

    cv::Mat gray;
    if (in.channels() == 3)
    {

      cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
      gray = in.clone();
    }

    cv::Mat gray_float;
    gray.convertTo(gray_float, CV_32F, 1.0f / 255.0f);

    cv::Mat laplacian_kernel = (cv::Mat_<float>(3, 3) << 
      0,  1, 0,
      1, -4, 1,
      0,  1, 0);
    cv::Mat edges;
    cv::filter2D(gray_float, edges, CV_32F, laplacian_kernel);

    cv::Mat abs_edges;
    cv::absdiff(edges, cv::Scalar(0), abs_edges);
    cv::sqrt(abs_edges, edges);

    double min_val, max_val;
    cv::minMaxLoc(edges, &min_val, &max_val);
    cv::Mat normalized;
    if (max_val > min_val)
    {
      edges = (edges - min_val) / (max_val - min_val + 1e-5);
    }
    else
    {
      edges = cv::Mat::zeros(edges.size(), CV_32F);
    }

    edges.convertTo(out, CV_8U, 255.0);
    return out;
  }

  cv::Mat ImageFusion::fusion(cv::Mat &eo, cv::Mat &ir)
  {

    cv::Mat boder, shadow;
    cv::Mat out;

    if (param_.edge_border > 1)
      cv::dilate(eo, boder, param_.bdStruct);
    else
      boder = eo;

    if (param_.do_shadow)
    {
      cv::dilate(boder, shadow, param_.sdStruct);
      cv::cvtColor(shadow, shadow, cv::COLOR_GRAY2BGR);
    }

    cv::Mat eo_3ch, ir_3ch;
    cv::cvtColor(boder, eo_3ch, cv::COLOR_GRAY2BGR);
    if (ir.channels() == 1) {
      cv::cvtColor(ir, ir_3ch, cv::COLOR_GRAY2BGR);
    } else {
      ir_3ch = ir.clone();
    }

    cv::Mat eo_enhanced;
    eo_3ch.convertTo(eo_enhanced, -1, 1.8, 0);

    cv::addWeighted(ir_3ch, 0.6, eo_enhanced, 1.0, 0, out);

    if (param_.do_shadow)
    {
      cv::Mat shadow_effect;
      shadow.convertTo(shadow_effect, -1, 0.1, 0);
      cv::subtract(out, shadow_effect, out);
    }

    return out;
  }
}

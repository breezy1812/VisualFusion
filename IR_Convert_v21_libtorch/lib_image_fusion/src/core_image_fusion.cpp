/*
 * core_image_fusuion.cpp
 *
 *  Created on: Feb 15, 2024
 *      Author: arthurho
 *
 *  Modified on: Feb 22, 2024
 *      Author: HongKai
 *
 * Modified on: Feb 29, 2024
 *      Author: HongKai
 *
 * Modified on: Mar 8, 2024
 *      Author: HongKai
 *
 * Modified on: Mar 17, 2024
 *      Author: HongKai
 */

#include <core_image_fusion.h>

namespace core
{

  ImageFusion::ImageFusion(Param param) : param_(std::move(param)) {}

  cv::Mat ImageFusion::equalization(cv::Mat &in)
  {
    cv::Mat out = in.clone();

    // 統計出現次數
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
    // MODIFIED: 按照Python TorchVerPreprocess實現邊緣檢測
    // Python code: 
    // gray = 0.299 * r + 0.587 * g + 0.114 * b
    // edges = F.conv2d(gray, laplacian_kernel, padding=1)
    // edges = edges.abs().sqrt()
    // edges = (edges - min_v) / (max_v - min_v + 1e-5)
    
    cv::Mat out;
    
    // 1. 確保輸入是灰度圖像
    cv::Mat gray;
    if (in.channels() == 3)
    {
      // 使用與Python相同的轉換係數
      cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
      gray = in.clone();
    }
    
    // 2. 轉換為32位浮點數
    cv::Mat gray_float;
    gray.convertTo(gray_float, CV_32F, 1.0f / 255.0f);
    
    // 3. Laplacian邊緣檢測 (與Python TorchVerPreprocess相同)
    // Python: laplacian_kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    cv::Mat laplacian_kernel = (cv::Mat_<float>(3, 3) << 
      0,  1, 0,
      1, -4, 1,
      0,  1, 0);
    
    cv::Mat edges;
    cv::filter2D(gray_float, edges, CV_32F, laplacian_kernel);
    
    // 4. 取絕對值並開根號 (與Python相同: edges.abs().sqrt())
    cv::Mat abs_edges;
    cv::absdiff(edges, cv::Scalar(0), abs_edges);
    cv::sqrt(abs_edges, edges);
    
    // 5. 正規化到[0,1] (與Python相同)
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
    
    // 6. 轉換回8位圖像格式
    edges.convertTo(out, CV_8U, 255.0);
    
    return out;
    
    /*
    // 原始程式碼保留作為註解：
    cv::Mat out;
    
    // 1. Gaussian blur (與Python相同)
    cv::Mat blur;
    cv::GaussianBlur(in, blur, cv::Size(5, 5), 0);
    
    // 2. Sobel gradients (與Python相同)
    cv::Mat sobel_x, sobel_y;
    cv::Sobel(blur, sobel_x, CV_32F, 1, 0, 3);
    cv::Sobel(blur, sobel_y, CV_32F, 0, 1, 3);
    
    // 3. Edge_x and edge_y with border roll logic (簡化版本)
    int border = 2;
    cv::Mat edge_x = cv::Mat::zeros(sobel_x.size(), CV_32F);
    cv::Mat edge_y = cv::Mat::zeros(sobel_y.size(), CV_32F);
    
    // 模擬Python的numpy.roll操作
    // Python: edge_x = np.where(sobel_x < 1.0, np.roll(sobel_x, border, axis=1), edge_x)
    cv::Mat mask_x_lt, mask_x_gt, mask_y_lt, mask_y_gt;
    cv::compare(sobel_x, 1.0, mask_x_lt, cv::CMP_LT);
    cv::compare(sobel_x, 1.0, mask_x_gt, cv::CMP_GT);
    cv::compare(sobel_y, 1.0, mask_y_lt, cv::CMP_LT);
    cv::compare(sobel_y, 1.0, mask_y_gt, cv::CMP_GT);
    
    // 簡化的roll操作實作
    cv::Mat shifted_x_pos, shifted_x_neg, shifted_y_pos, shifted_y_neg;
    
    // X方向的shift
    cv::Mat M_x = (cv::Mat_<float>(2, 3) << 1, 0, border, 0, 1, 0);
    cv::warpAffine(sobel_x, shifted_x_pos, M_x, sobel_x.size(), cv::INTER_LINEAR, cv::BORDER_WRAP);
    
    M_x = (cv::Mat_<float>(2, 3) << 1, 0, -border, 0, 1, 0);
    cv::warpAffine(-sobel_x, shifted_x_neg, M_x, sobel_x.size(), cv::INTER_LINEAR, cv::BORDER_WRAP);
    
    // Y方向的shift
    cv::Mat M_y = (cv::Mat_<float>(2, 3) << 1, 0, 0, 0, 1, border);
    cv::warpAffine(sobel_y, shifted_y_pos, M_y, sobel_y.size(), cv::INTER_LINEAR, cv::BORDER_WRAP);
    
    M_y = (cv::Mat_<float>(2, 3) << 1, 0, 0, 0, 1, -border);
    cv::warpAffine(-sobel_y, shifted_y_neg, M_y, sobel_y.size(), cv::INTER_LINEAR, cv::BORDER_WRAP);
    
    // 應用mask
    shifted_x_pos.copyTo(edge_x, mask_x_lt);
    shifted_x_neg.copyTo(edge_x, mask_x_gt);
    shifted_y_pos.copyTo(edge_y, mask_y_lt);
    shifted_y_neg.copyTo(edge_y, mask_y_gt);
    
    // 4. Edge magnitude
    cv::Mat edge;
    cv::magnitude(edge_x, edge_y, edge);
    
    // 5. Sobel magnitude
    cv::Mat sobel;
    cv::magnitude(sobel_x, sobel_y, sobel);
    
    // 6. Final edge (與Python相同: edge = sobel - edge)
    cv::subtract(sobel, edge, out);
    
    // 7. Clip to valid range (與Python相同: np.clip(edge, -255, 255))
    cv::Mat clipped;
    cv::max(out, -255.0, clipped);
    cv::min(clipped, 255.0, out);
    
    // 8. 轉換為8位圖像並重複到3通道 (與Python相同)
    out.convertTo(out, CV_8U, 1.0, 128.0); // 將[-255,255]映射到[0,255]
    
    return out;
    */
  }

  cv::Mat ImageFusion::fusion(cv::Mat &eo, cv::Mat &ir)
  {
    // 恢復原始程式碼：支持邊緣疊加和陰影處理
    cv::Mat boder, shadow;
    cv::Mat out = ir.clone();

    if (param_.edge_border > 1)
      cv::dilate(eo, boder, param_.bdStruct);
    else
      boder = eo;

    if (param_.do_shadow)
    {
      cv::dilate(boder, shadow, param_.sdStruct);
      cv::cvtColor(shadow, shadow, cv::COLOR_GRAY2BGR);
      cv::subtract(out, shadow, out);
    }

    cv::cvtColor(boder, boder, cv::COLOR_GRAY2BGR);
    cv::add(out, boder, out);

    return out;
  }
} /* namespace core */

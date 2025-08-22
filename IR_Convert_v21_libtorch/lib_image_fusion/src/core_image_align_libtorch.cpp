#include <core_image_align_libtorch.h>
#include "util_timer.h"

namespace core
{
  ImageAlign::ImageAlign(Param param) : param_(std::move(param))
  {
    torch::manual_seed(1);
    torch::autograd::GradMode::set_enabled(false);

    if (param_.device.compare("cuda") == 0 && torch::cuda::is_available())
    {
      torch::Device cuda(torch::kCUDA);
      device = cuda;
    }

    net = torch::jit::load(param_.model_path);
    net.eval();
    net.to(device);

    if (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0)
      net.to(torch::kHalf);

    // if (param_.device.compare("cuda") == 0)
    //   warm_up();
  }

  // warm up
  void ImageAlign::warm_up()
  {
    printf("Warm up...\n");

    cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;
    cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 255;

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; i++)
    {
      cv::Mat H;
      std::vector<cv::Point2i> eo_mkpts, ir_mkpts;
      pred(eo, ir, eo_mkpts, ir_mkpts);
    }

    const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
    const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    printf("Warm up done in %.2f s\n", period);
  }

  // prediction - MODIFIED: 改善精度和資料處理，符合Python版本
  void ImageAlign::pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts)
  {
    if (eo.channels() != 1 || ir.channels() != 1)
      throw std::runtime_error("ImageAlign::pred: eo and ir must be single channel images");

    // resize input image to pred_width x pred_height
    cv::Mat eo_temp, ir_temp;
    cv::resize(eo, eo_temp, cv::Size(param_.pred_width, param_.pred_height));
    cv::resize(ir, ir_temp, cv::Size(param_.pred_width, param_.pred_height));

    // MODIFIED: 改善正規化，避免FP16精度損失，與Python版本保持一致
    // 將圖像轉換為float32並正規化到[0,1]
    cv::Mat eo_float, ir_float;
    eo_temp.convertTo(eo_float, CV_32F, 1.0f / 255.0f);
    ir_temp.convertTo(ir_float, CV_32F, 1.0f / 255.0f);
    
    // 確保記憶體連續性，與Python版本一致
    eo_float = eo_float.clone();
    ir_float = ir_float.clone();

    // 創建tensor，避免from_blob可能的記憶體問題
    torch::Tensor eo_tensor = torch::from_blob(eo_float.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).clone().to(device);
    torch::Tensor ir_tensor = torch::from_blob(ir_float.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).clone().to(device);
    
    // MODIFIED: 只在明確需要時才使用FP16，優先保持FP32精度
    if (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0)
    {
      eo_tensor = eo_tensor.to(torch::kHalf);
      ir_tensor = ir_tensor.to(torch::kHalf);
    }

    // run the model
    torch::IValue pred = net.forward({eo_tensor, ir_tensor});
    torch::jit::Stack pred_ = pred.toTuple()->elements();

    // get mkpts from the model output
    torch::Tensor eo_mkpts = pred_[0].toTensor().to(torch::kFloat32); // 確保輸出是FP32
    torch::Tensor ir_mkpts = pred_[1].toTensor().to(torch::kFloat32);
    int leng=pred_[2].toInt(); // 獲取特徵點數量

    // clean up eo_pts and ir_pts
    eo_pts.clear();
    ir_pts.clear();

    for (int i = 0; i <leng; i++)
    {
      // 使用round而非直接轉換，提高精度
      float eo_x = eo_mkpts[i][0].item<float>();
      float eo_y = eo_mkpts[i][1].item<float>();
      float ir_x = ir_mkpts[i][0].item<float>();
      float ir_y = ir_mkpts[i][1].item<float>();
      
      eo_pts.push_back(cv::Point2i(static_cast<int>(std::round(eo_x)), static_cast<int>(std::round(eo_y))));
      ir_pts.push_back(cv::Point2i(static_cast<int>(std::round(ir_x)), static_cast<int>(std::round(ir_y))));
    }
    // std::cout <<"leng="<<leng<<"  - eo_pts coordinates:" << std::endl;
    // for (size_t i = 0; i < leng; ++i) {
    //   std::cout << "    [" << i << "]: (" << eo_pts[i].x << ", " << eo_pts[i].y << ")" << std::endl;
    // }

    // DEBUG: 輸出特徵點數量
    std::cout << "  - Model extracted " << eo_pts.size() << " feature point pairs" << std::endl;
  }

  // align with last H - MODIFIED: 簡化過濾邏輯，使用Python風格的直接輸出
  void ImageAlign::align(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, cv::Mat &H)
  {
    // predict keypoints
    pred(eo, ir, eo_pts, ir_pts);

    // 只進行基本的座標縮放和偏移調整
    if (param_.out_width_scale - 1 > 1e-6 || param_.out_height_scale - 1 > 1e-6 || param_.bias_x > 0 || param_.bias_y > 0)
    {
      for (cv::Point2i &i : eo_pts)
      {
        i.x = i.x * param_.out_width_scale + param_.bias_x;
        i.y = i.y * param_.out_height_scale + param_.bias_y;
      }
      for (cv::Point2i &i : ir_pts)
      {
        i.x = i.x * param_.out_width_scale + param_.bias_x;
        i.y = i.y * param_.out_height_scale + param_.bias_y;
      }
    }
    
    // 輸出最終特徵點數量
    std::cout << "  - Final feature points after coordinate adjustment: " << eo_pts.size() << std::endl;
  }
} /* namespace core */

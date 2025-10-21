#include <core_image_align_libtorch.h>
#include "util_timer.h"
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <experimental/filesystem>

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

    printf("Model initialization completed\n");

    if (param_.device.compare("cuda") == 0) {
      printf("Performing smart warmup to initialize CUDA kernels...\n");
      smart_warmup();
    }
  }

  void ImageAlign::smart_warmup()
  {
    printf("Smart warmup for CUDA kernel initialization...\n");

    cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
    cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;

    const auto t0 = std::chrono::high_resolution_clock::now();

    torch::Tensor eo_tensor = torch::from_blob(eo.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).clone().to(device).to(torch::kFloat32) / 255.0f;
    torch::Tensor ir_tensor = torch::from_blob(ir.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).clone().to(device).to(torch::kFloat32) / 255.0f;
    bool use_fp16 = (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0);
    if (use_fp16) {
      eo_tensor = eo_tensor.to(torch::kHalf);
      ir_tensor = ir_tensor.to(torch::kHalf);
    }

    torch::IValue pred = net.forward({eo_tensor, ir_tensor});

    printf("Reloading model to maintain first-inference precision...\n");
    net = torch::jit::load(param_.model_path);
    net.eval();
    net.to(device);
    if (use_fp16) {
      net.to(torch::kHalf);
    }

    const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
    const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    printf("Smart warmup completed in %.2f s\n", period);
  }

  void ImageAlign::pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, const std::string& filename)
  {
    if (eo.channels() != 1 || ir.channels() != 1)
      throw std::runtime_error("ImageAlign::pred: eo and ir must be single channel images");

    cv::Mat eo_float, ir_float;
    eo.convertTo(eo_float, CV_32F, 1.0 / 255.0, 0.0);
    ir.convertTo(ir_float, CV_32F, 1.0 / 255.0, 0.0);

    torch::Tensor eo_tensor = torch::from_blob(eo_float.ptr<float>(), {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).clone().to(device);
    torch::Tensor ir_tensor = torch::from_blob(ir_float.ptr<float>(), {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).clone().to(device);

    auto model_inference_start = std::chrono::high_resolution_clock::now();

    torch::IValue pred = net.forward({eo_tensor, ir_tensor});

    auto model_inference_end = std::chrono::high_resolution_clock::now();
    double model_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(model_inference_end - model_inference_start).count() / 1000000.0;
    torch::jit::Stack pred_ = pred.toTuple()->elements();

    torch::Tensor feat_vi = pred_[0].toTensor();
    torch::Tensor feat_ir = pred_[1].toTensor();
    if(!filename.empty()){
      std::string feat_csv_filename = "../../../IR_Convert_v21_libtorch/output/feat_data_"+filename + ".csv";
      std::ofstream feat_csv(feat_csv_filename);
      if (feat_csv.is_open()) {

        std::cout<<"debug: Saving feature channels to CSV: " << feat_csv_filename << std::endl;
        torch::Tensor eo_flat = feat_vi.detach().cpu().flatten();
        torch::Tensor ir_flat = feat_ir.detach().cpu().flatten();
        int n = eo_flat.size(0);
        feat_csv << std::fixed << std::setprecision(20);
        for (int i = 0; i < n; ++i) {
          float vi_val = eo_flat[i].item<float>();
          float ir_val = ir_flat[i].item<float>();
          feat_csv << vi_val << "," << ir_val << "\n";
        }

        feat_csv.close();
      }
    }

    eo_pts.clear();
    ir_pts.clear();

    std::cout << "Model now outputs bias and weight instead of keypoints." << std::endl;
    std::cout << "No keypoints extracted." << std::endl;

    std::cout << "Model now outputs bias and weight. No keypoints processing available." << std::endl;

    writeTimingToCSV("Model_Inference", model_inference_time, 0, filename);
    printf("Model inference time: %.6f s\n", model_inference_time);

    std::cout << "  - Model extracted 0 feature point pairs (bias/weight mode)" << std::endl;
  }

  void ImageAlign::align(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, cv::Mat &H, const std::string& filename)
  {

    pred(eo, ir, eo_pts, ir_pts, filename);

    if (std::abs(param_.out_width_scale - 1.0) > 1e-6 || std::abs(param_.out_height_scale - 1.0) > 1e-6 || param_.bias_x > 0 || param_.bias_y > 0)
    {
      for (cv::Point2i &i : eo_pts)
      {
        i.x = i.x * param_.out_width_scale  ;
        i.y = i.y * param_.out_height_scale ;
      }
      for (cv::Point2i &i : ir_pts)
      {
        i.x = i.x * param_.out_width_scale  ;
        i.y = i.y * param_.out_height_scale ;
      }
      std::cout << "  - Feature point scaling applied: pred(" << param_.pred_width << "x" << param_.pred_height 
                << ") -> out(" << param_.out_width_scale * param_.pred_width << "x" << param_.out_height_scale * param_.pred_height 
                << "), scale=(" << param_.out_width_scale << ", " << param_.out_height_scale 
                << "), bias=(" << param_.bias_x << ", " << param_.bias_y << ")" << std::endl;
    }
    else
    {
      std::cout << "  - No feature point scaling needed: scale=(" << param_.out_width_scale << ", " << param_.out_height_scale 
                << "), bias=(" << param_.bias_x << ", " << param_.bias_y << ")" << std::endl;
    }

    std::cout << "  - Final feature points after coordinate adjustment: " << eo_pts.size() << std::endl;
  }

  void ImageAlign::writeTimingToCSV(const std::string& operation, double time_ms, int leng, const std::string& filename)
  {
    std::string csv_filename = "./itiming_log.csv";
    bool file_exists = std::experimental::filesystem::exists(csv_filename);
    std::ofstream csv_file(csv_filename, std::ios::app);
    if (!file_exists) {

      csv_file << "Filename,Operation,Time_s,Mode,keypoints\n";
    }

    std::string identifier;
    if (!filename.empty()) {
      identifier = filename;
    } else {
      std::cout<<"warm up"<<std::endl;
      return;
    }

    csv_file << identifier << "," 
             << operation << "," 
             << std::fixed << std::setprecision(6) << time_ms << ","
             << param_.mode << ","
             << leng << "\n";
    csv_file.close();
  }
} 

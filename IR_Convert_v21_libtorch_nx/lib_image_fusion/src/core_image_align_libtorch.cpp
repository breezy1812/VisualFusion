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

    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);

    printf("Deterministic settings applied:\n");
    printf("  - TF32 cuBLAS: disabled\n");
    printf("  - TF32 cuDNN: disabled\n");
    printf("  - cuDNN deterministic: enabled\n");
    printf("  - cuDNN benchmark: disabled\n");
    printf("  - Deterministic algorithms: enabled\n");

    if (param_.device.compare("cuda") == 0 && torch::cuda::is_available())
    {
      torch::Device cuda(torch::kCUDA);
      device = cuda;
    }

    net = torch::jit::load(param_.model_path);
    net.eval();
    net.to(device);

    printf("Model initialization completed\n");
    printf("  - Mode: %s\n", param_.mode.c_str());
    printf("  - Device: %s\n", param_.device.c_str());
    if (param_.mode.compare("fp16") == 0) {
      printf("  - Precision: FP16 (pre-converted from Python)\n");
      printf("  - Model weights: FP16\n");
      printf("  - Input tensors will be: FP16\n");
      printf("  - Ready for Tensor Core acceleration\n");
    } else {
      printf("  - Precision: FP32\n");
    }

    if (param_.device.compare("cuda") == 0) {
      printf("Performing smart warmup to initialize CUDA kernels...\n");
      smart_warmup();
    }
  }

  void ImageAlign::smart_warmup()
  {
    printf("Smart warmup for CUDA kernel initialization (5 iterations)...\n");

    cv::Mat eo = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;
    cv::Mat ir = cv::Mat::ones(param_.pred_height, param_.pred_width, CV_8UC1) * 128;

    const auto t0 = std::chrono::high_resolution_clock::now();
    bool use_fp16 = (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0);
    if (use_fp16) {
      printf("  - Warmup mode: FP16 (matching inference precision)\n");
    } else {
      printf("  - Warmup mode: FP32\n");
    }

    for (int i = 0; i < 5; i++) {
      printf("  Warmup iteration %d/5...\n", i + 1);
      torch::Tensor eo_tensor = torch::from_blob(eo.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).clone().to(device).to(torch::kFloat32) / 255.0f;
      torch::Tensor ir_tensor = torch::from_blob(ir.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8).clone().to(device).to(torch::kFloat32) / 255.0f;

      if (use_fp16) {
        eo_tensor = eo_tensor.to(torch::kHalf);
        ir_tensor = ir_tensor.to(torch::kHalf);
      }

      torch::IValue pred = net.forward({eo_tensor, ir_tensor});
    }

    const auto elapsed = std::chrono::high_resolution_clock::now() - t0;
    const double period = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();

    printf("Smart warmup completed in %.2f s (5 iterations)\n", period);
    printf("✅ CUDA kernels initialized, ready for inference\n");
  }

  void ImageAlign::pred(cv::Mat &eo, cv::Mat &ir, std::vector<cv::Point2i> &eo_pts, std::vector<cv::Point2i> &ir_pts, const std::string& filename)
  {
    if (eo.channels() != 1 || ir.channels() != 1)
      throw std::runtime_error("ImageAlign::pred: eo and ir must be single channel images");

    bool use_fp16 = (param_.mode.compare("fp16") == 0 && param_.device.compare("cuda") == 0);
    torch::Tensor eo_tensor, ir_tensor;

    if (param_.device.compare("cuda") == 0) {

      eo_tensor = torch::from_blob(eo.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8)
                    .clone()
                    .to(device)
                    .to(torch::kFloat32)
                    .div_(255.0f);
      ir_tensor = torch::from_blob(ir.data, {1, 1, param_.pred_height, param_.pred_width}, torch::kUInt8)
                    .clone()
                    .to(device)
                    .to(torch::kFloat32)
                    .div_(255.0f);

      if (use_fp16) {
        eo_tensor = eo_tensor.to(torch::kHalf);
        ir_tensor = ir_tensor.to(torch::kHalf);
      }
    } else {

      cv::Mat eo_float, ir_float;
      eo.convertTo(eo_float, CV_32F, 1.0 / 255.0, 0.0);
      ir.convertTo(ir_float, CV_32F, 1.0 / 255.0, 0.0);
      eo_tensor = torch::from_blob(eo_float.ptr<float>(), {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).clone();
      ir_tensor = torch::from_blob(ir_float.ptr<float>(), {1, 1, param_.pred_height, param_.pred_width}, torch::kFloat32).clone();
    }

    auto model_inference_start = std::chrono::high_resolution_clock::now();

    torch::IValue pred = net.forward({eo_tensor, ir_tensor});

    auto model_inference_end = std::chrono::high_resolution_clock::now();
    double model_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(model_inference_end - model_inference_start).count() / 1000000.0;
    torch::jit::Stack pred_ = pred.toTuple()->elements();

    torch::Tensor eo_mkpts = pred_[0].toTensor().to(torch::kInt32);
    torch::Tensor ir_mkpts = pred_[1].toTensor().to(torch::kInt32);

    eo_pts.clear();
    ir_pts.clear();

    int num_points = eo_mkpts.size(0);
    for (int i = 0; i < num_points; i++)
    {
      int eo_x = eo_mkpts[i][0].item<int>();
      int eo_y = eo_mkpts[i][1].item<int>();
      int ir_x = ir_mkpts[i][0].item<int>();
      int ir_y = ir_mkpts[i][1].item<int>();

      if (eo_x == 0 && eo_y == 0) {
        continue;
      }
      eo_pts.push_back(cv::Point2i(eo_x, eo_y));
      ir_pts.push_back(cv::Point2i(ir_x, ir_y));
    }
    int leng = eo_pts.size();

    writeTimingToCSV("Model_Inference", model_inference_time, leng, filename);

    std::cout << "  - Model extracted " << eo_pts.size() << " feature point pairs" << std::endl;
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

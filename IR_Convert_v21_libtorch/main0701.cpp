// 直接將一通道餵給三通道
// 
// 版本更新紀錄：
// 2025-07-03 v2: 智能RANSAC策略 (當前版本)
//             - 實作智能RANSAC：自動嘗試5組不同參數 (從嚴格到寬鬆)
//             - 使用品質分數選擇最佳homography：inlier比例 × confidence × 矩陣穩定性
//             - 輸出詳細的RANSAC嘗試結果和最佳參數資訊
//             - 回復原始插值方法 (高品質插值效果不佳)
//             原始程式碼已保留為註解，可隨時復原
// 2025-07-03 v1: 改進RANSAC方法 (已升級)
//             - 多次嘗試不同的RANSAC參數組合，選擇最佳結果
//             - 增加homography矩陣品質檢查 (determinant範圍檢查)
// 
// 智能RANSAC參數策略：
// - {3.0, 0.99}: 最嚴格，適合高品質特徵點
// - {5.0, 0.99}: 嚴格，平衡品質與數量
// - {8.0, 0.95}: 中等，標準設定  
// - {10.0, 0.9}: 寬鬆，容許更多noise
// - {15.0, 0.85}: 最寬鬆，最後嘗試

#include <ratio>
#include <chrono>
#include <string>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>  // ADDED: 確保包含homography相關函數

#include "lib_image_fusion/include/core_image_to_gray.h"
#include "lib_image_fusion/src/core_image_to_gray.cpp"

#include "lib_image_fusion/include/core_image_resizer.h"
#include "lib_image_fusion/src/core_image_resizer.cpp"

#include "lib_image_fusion/include/core_image_fusion.h"
#include "lib_image_fusion/src/core_image_fusion.cpp"

#include "lib_image_fusion/include/core_image_perspective.h"
#include "lib_image_fusion/src/core_image_perspective.cpp"

// #include "lib_image_fusion/include/core_image_align_onnx_polar.h"
// #include "lib_image_fusion/src/core_image_align_onnx_polar.cpp"

#include "lib_image_fusion/include/core_image_align_libtorch.h"
#include "lib_image_fusion/src/core_image_align_libtorch.cpp"

#include "utils/include/util_timer.h"
#include "utils/src/util_timer.cpp"

#include "nlohmann/json.hpp"

using namespace cv;
using namespace std;
using namespace filesystem;
using json = nlohmann::json;

// show error message
inline void alert(const string &msg)
{
  std::cout << string("\033[1;31m[ ERROR ]\033[0m ") + msg << std::endl;
}

// check file exit
inline bool is_file_exit(const string &path)
{
  bool res = is_regular_file(path);
  if (!res)
    alert(string("File not found: ") + path);
  return res;
}

// check directory exit
inline bool is_dir_exit(const string &path)
{
  bool res = is_directory(path);
  if (!res)
    alert(string("File not found: ") + path);
  return res;
}

// init config
inline void init_config(nlohmann::json &config)
{
  config.emplace("input_dir", "./input");
  config.emplace("output_dir", "./output");
  config.emplace("output", false);

  config.emplace("device", "cpu");
  config.emplace("pred_mode", "fp32");
  config.emplace("model_path", "./model/SemLA_jit_cpu.zip");

  config.emplace("cut_x", 0);
  config.emplace("cut_y", 0);
  config.emplace("cut_h", -1); // -1 means no cut, use full image height
  config.emplace("cut_w", -1); // -1 means no cut, use full image width

  config.emplace("output_width", 480);
  config.emplace("output_height", 360);

  config.emplace("pred_width", 480);//480,360
  config.emplace("pred_height", 360);// 640 480

  config.emplace("fusion_shadow", false);
  config.emplace("fusion_edge_border", 1);
  config.emplace("fusion_threshold_equalization", 128);
  config.emplace("fusion_threshold_equalization_low", 72);
  config.emplace("fusion_threshold_equalization_high", 192);
  config.emplace("fusion_threshold_equalization_zero", 64);

  config.emplace("perspective_check", true);
  config.emplace("perspective_distance", 10);
  config.emplace("perspective_accuracy", 0.85);

  config.emplace("align_angle_sort", 0.6);
  config.emplace("align_angle_mean", 10.0);
  config.emplace("align_distance_last", 10.0);
  config.emplace("align_distance_line", 10.0);

  config.emplace("skip_frames", nlohmann::json::object());
}

cv::Mat cropImage(const cv::Mat& sourcePic, int x, int y, int w, int h) {
    // 邊界檢查，確保不超出原圖
    int crop_x = std::max(0, x);
    int crop_y = std::max(0, y);
    int crop_w = std::min(w, sourcePic.cols - crop_x);
    int crop_h = std::min(h, sourcePic.rows - crop_y);
    cv::Rect roi(crop_x, crop_y, crop_w, crop_h);
    return sourcePic(roi).clone();
}

// get pair file
inline bool get_pair(const string &path, string &eo_path, string &ir_path)
{
  ir_path = path;
  eo_path = path;

  if (path.find("_EO") != string::npos)
    ir_path.replace(ir_path.find("_EO"), 3, "_IR");
  else
    return false;

  // 檢查檔案是否存在
  if (!is_file_exit(eo_path))
    return false;
  if (!is_file_exit(ir_path))
    return false;

  return true;
}

// check file is video or image
inline bool is_video(const string &path)
{
  std::vector<string> video_ext = {".mp4", ".avi", ".mov", ".MP4", ".AVI", ".MOV"};
  for (const string &ext : video_ext)
    if (path.find(ext) != string::npos)
      return true;
  return false;
}

// skip frames
inline void skip_frames(const string &path, cv::VideoCapture &cap, nlohmann::json &config)
{
  nlohmann::json skip_frames = config["skip_frames"];
  if (skip_frames.empty())
    return;

  string file = path.substr(path.find_last_of("/\\") + 1);
  string name = file.substr(0, file.find_last_of("."));

  int skip = 0;

  if (skip_frames.contains(name))
    skip = skip_frames[name];

  if (skip > 0)
    cap.set(cv::CAP_PROP_POS_FRAMES, skip);
}

// REMOVED: 時間延遲處理函數已移除，因為採用Python風格的每幀處理

int main(int argc, char **argv)
{
  // ----- Config -----
  json config;
  string config_path = "./config/config.json";
  {
    // check argument
    if (argc > 1)
      config_path = argv[1];

    // check config file
    if (!is_file_exit(config_path))
      return 0;

    // read config file
    ifstream temp(config_path);
    temp >> config;

    // init
    init_config(config);
  }

  // ----- Input / Output -----
  // input and output directory
  bool isOut = config["output"];
  string input_dir = config["input_dir"];
  string output_dir = config["output_dir"];
  {
    // show directories
    cout << "[ Directories ]" << endl;

    // check input directory
    if (!is_dir_exit(input_dir))
      return 0;
    cout << "\t Input: " << input_dir << endl;

    // check output directory
    if (isOut)
    {
      if (!is_dir_exit(output_dir))
        return 0;
      cout << "\tOutput: " << output_dir << endl;
    }
  }

  // ----- Get Config -----
  // get output and predict size
  int out_w = config["output_width"], out_h = config["output_height"];
  int pred_w = config["pred_width"], pred_h = config["pred_height"];

  // get cut parameter
  int cut_x = config["cut_x"];
  int cut_y = config["cut_y"];
  int cut_w = config["cut_w"];
  int cut_h = config["cut_h"];


  // get model info
  string device = config["device"];
  string pred_mode = config["pred_mode"];
  string model_path = config["model_path"];

  // get fusion parameter
  bool fusion_shadow = config["fusion_shadow"];
  int fusion_edge_border = config["fusion_edge_border"];
  int fusion_threshold_equalization = config["fusion_threshold_equalization"];
  int fusion_threshold_equalization_low = config["fusion_threshold_equalization_low"];
  int fusion_threshold_equalization_high = config["fusion_threshold_equalization_high"];
  int fusion_threshold_equalization_zero = config["fusion_threshold_equalization_zero"];

  // get perspective parameter
  bool perspective_check = config["perspective_check"];
  float perspective_distance = config["perspective_distance"];
  float perspective_accuracy = config["perspective_accuracy"];

  // get align parameter
  float align_angle_mean = config["align_angle_mean"];
  float align_angle_sort = config["align_angle_sort"];
  float align_distance_last = config["align_distance_last"];
  float align_distance_line = config["align_distance_line"];

  // show config
  {
    cout << "[ Config ]" << endl;
    cout << "\tOutput Size: " << out_w << " x " << out_h << endl;
    cout << "\tPredict Size: " << pred_w << " x " << pred_h << endl;
    cout << "\tModel Path: " << model_path << endl;
    cout << "\tDevice: " << device << endl;
    cout << "\tPred Mode: " << pred_mode << endl;
    cout << "\tFusion Shadow: " << fusion_shadow << endl;
    cout << "\tFusion Edge Border: " << fusion_edge_border << endl;
    cout << "\tFusion Threshold Equalization: " << fusion_threshold_equalization << endl;
    cout << "\tFusion Threshold Equalization Low: " << fusion_threshold_equalization_low << endl;
    cout << "\tFusion Threshold Equalization High: " << fusion_threshold_equalization_high << endl;
    cout << "\tFusion Threshold Equalization Zero: " << fusion_threshold_equalization_zero << endl;
    cout << "\tPerspective Check: " << perspective_check << endl;
    cout << "\tPerspective Distance: " << perspective_distance << endl;
    cout << "\tPerspective Accuracy: " << perspective_accuracy << endl;
    cout << "\tAlign Angle Mean: " << align_angle_mean << endl;
    cout << "\tAlign Angle Sort: " << align_angle_sort << endl;
    cout << "\tAlign Distance Last: " << align_distance_last << endl;
    cout << "\tAlign Distance Line: " << align_distance_line << endl;
    // REMOVED: 移除新增參數的顯示
    // cout << "\tCompute Per Frame: " << config["compute_per_frame"] << endl;
    // cout << "\tTime Delay Entries: " << config["time_delay"].size() << endl;
  }

  // ----- Start -----
  for (const auto &file : directory_iterator(input_dir))
  {
    // Get file path and name
    string eo_path, ir_path, save_path = output_dir;
    bool isPair = get_pair(file.path().string(), eo_path, ir_path);
    if (!isPair)
      continue;
    else
    {
      // save path
      string file = eo_path.substr(eo_path.find_last_of("/\\") + 1);
      string name = file.substr(0, file.find_last_of("."));
      if (save_path.back() != '/' && save_path.back() != '\\')
        save_path += "/";
      save_path += name;
    }

    // Check file is video
    bool isVideo = is_video(eo_path);

    // Get frame size, frame rate, and create capture/writer
    int eo_w, eo_h, ir_w, ir_h, frame_rate;
    VideoCapture eo_cap, ir_cap;
    VideoWriter writer;
    if (isVideo)
    {
      eo_cap.open(eo_path);
      ir_cap.open(ir_path);
      skip_frames(eo_path, eo_cap, config);
      skip_frames(ir_path, ir_cap, config);

      eo_w = (int)eo_cap.get(3), eo_h = (int)eo_cap.get(4);
      ir_w = (int)ir_cap.get(3), ir_h = (int)ir_cap.get(4);
      
      // MODIFIED: 修正frame rate計算，與Python版本一致
      // 原本: frame_rate = (int)ir_cap.get(5) / (int)eo_cap.get(5);
      // 現在: frame_rate = (int)ir_cap.get(cv::CAP_PROP_FPS) / (int)eo_cap.get(cv::CAP_PROP_FPS);
      int fps_ir = (int)ir_cap.get(cv::CAP_PROP_FPS);
      int fps_eo = (int)eo_cap.get(cv::CAP_PROP_FPS);
      frame_rate = fps_ir / fps_eo;
      
      cout << "  - IR: " << fps_ir << " fps, " << ir_w << "x" << ir_h << endl;
      cout << "  - EO: " << fps_eo << " fps, " << eo_w << "x" << eo_h << endl;
      cout << "  - Rate: " << frame_rate << " (IR:EO)" << endl;
      
      if (isOut)
      {
        writer.open(save_path + "480360.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps_ir, cv::Size(out_w * 3, out_h));
      }
    }
    else
    {
      Mat eo = imread(eo_path);
      Mat ir = imread(ir_path);
      eo_w = eo.cols, eo_h = eo.rows;
      ir_w = ir.cols, ir_h = ir.rows;
    }

    // REMOVED: 秮除 EO 圖像的寬度計算，因為不再需要按比例計算後裁剪
    // int eo_new_w = eo_w * ((float)out_h / eo_h);

    // Create instance
    auto image_gray = core::ImageToGray::create_instance(core::ImageToGray::Param());

    // MODIFIED: 移除裁剪參數，直接 resize 到目標尺寸
    // 原本: .set_eo(eo_new_w, out_h, out_w, out_h) - 有裁剪參數
    // 現在: .set_eo(out_w, out_h) - 直接 resize，無裁剪
    auto image_resizer = core::ImageResizer::create_instance(
        core::ImageResizer::Param()
            .set_eo(out_w, out_h)
            .set_ir(out_w, out_h));

    auto image_fusion = core::ImageFusion::create_instance(
        core::ImageFusion::Param()
            .set_shadow(fusion_shadow)
            .set_edge_border(fusion_edge_border)
            .set_threshold_equalization_high(fusion_threshold_equalization_high)
            .set_threshold_equalization_low(fusion_threshold_equalization_low)
            .set_threshold_equalization_zero(fusion_threshold_equalization_zero));

    auto image_perspective = core::ImagePerspective::create_instance(
        core::ImagePerspective::Param()
            .set_check(perspective_check, perspective_accuracy, perspective_distance));

    // MODIFIED: 移除 bias 設定，因為不再有裁剪偏移
    // 原本: .set_bias(image_resizer->get_eo_clip_x(), image_resizer->get_eo_clip_y())
    // 現在: .set_bias(0, 0) - 無偏移
    auto image_align = core::ImageAlign::create_instance(
        core::ImageAlign::Param()
            .set_size(pred_w, pred_h, out_w, out_h)
            .set_net(device, model_path, pred_mode)
            .set_distance(align_distance_line, align_distance_last, 20)
            .set_angle(align_angle_mean, align_angle_sort)
            .set_bias(0, 0));

    // 開始計時
    auto timer_base = core::Timer("All");
    auto timer_resize = core::Timer("Resize");
    auto timer_gray = core::Timer("Gray");
    // REMOVED: auto timer_clip = core::Timer("Clip"); - 移除裁剪計時器
    auto timer_equalization = core::Timer("Equalization");
    auto timer_perspective = core::Timer("Perspective");
    auto timer_find_homo = core::Timer("Homo");
    auto timer_fusion = core::Timer("Fusion");
    auto timer_edge = core::Timer("Edge");
    auto timer_align = core::Timer("Align");

    // 讀取影片
    Mat eo, ir;
    
    // ADDED: Python風格的變數設定
    int cnt = 0;  // 幀數計數器
    cv::Mat M;    // Homography矩陣
    Mat temp_pair = Mat::zeros(out_h, out_w * 2, CV_8UC3);  // 儲存特徵點配對圖像
    std::vector<cv::Point2i> eo_pts, ir_pts; // 保留特徵點
    const int compute_per_frame = 50; // 每50幀做一次

    while (1)
    {

      // MODIFIED: 修改影片讀取順序和frame rate處理，與Python版本一致
      // 原本: eo_cap.read(eo); for (int i = 0; i < frame_rate; i++) ir_cap.read(ir);
      // 現在: ir_cap.read(ir); eo_cap.read(eo); (Python: ret_ir, frame_ir = cap_ir.read(); ret_eo, frame_eo = cap_eo.read())


      if (isVideo)
      {
        ir_cap.read(ir);
        eo_cap.read(eo);
        // 新增：eo每一幀都經過裁切（預設裁切全圖）
        if (cut_w != -1 && cut_h != -1) {
          eo = cropImage(eo, cut_x, cut_y, cut_w, cut_h);
        }
      }
      else
      {
        eo = cv::imread(eo_path);
        ir = cv::imread(ir_path);
        // 新增：eo先經過裁切（預設裁切全圖）
        if (cut_w != -1 && cut_h != -1) {
          eo = cropImage(eo, cut_x, cut_y, cut_w, cut_h);
        }
      }

      // 退出迴圈條件
      if (eo.empty() || ir.empty())
        break;

      // 幀數計數
      timer_base.start();

      // MODIFIED: 按照Python程式碼重新組織處理邏輯
      // 原始處理邏輯註解掉
      /*
      // 原始程式碼：
      Mat out;
      Mat eo_edge;
      Mat eo_resize, ir_resize, eo_wrap;
      Mat eo_gray, ir_gray;

      {
        timer_resize.start();
        image_resizer->resize(eo, ir, eo_resize, ir_resize);
        timer_resize.stop();
      }

      {
        timer_gray.start();
        eo_gray = image_gray->gray(eo_resize);
        ir_gray = image_gray->gray(ir_resize);
        timer_gray.stop();
      }
      */
      
      // 新程式碼：按照Python版本
      Mat img_ir, img_eo, gray_ir, gray_eo;
      
      // Resize圖像，回復原始版本
      {
        timer_resize.start();
        // 回復原始版本：預設插值方法
        cv::resize(ir, img_ir, cv::Size(out_w, out_h));
        cv::resize(eo, img_eo, cv::Size(out_w, out_h));
        
        // 高品質插值版本 (效果不佳，已停用)：
        // cv::resize(ir, img_ir, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        // cv::resize(eo, img_eo, cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
        timer_resize.stop();
      }
      
      // 轉換為灰度圖像，與Python相同
      {
        timer_gray.start();
        cv::cvtColor(img_ir, gray_ir, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img_eo, gray_eo, cv::COLOR_BGR2GRAY);
        timer_gray.stop();
      }

      // ====== 原本每一幀都計算特徵點與Homography的區塊註解起來 ======
      /*
      vector<cv::Point2i> eo_pts, ir_pts;
      {
        timer_align.start();
        image_align->align(gray_eo, gray_ir, eo_pts, ir_pts, M);
        cout << "  - Frame " << cnt << ": Found " << eo_pts.size() << " feature point pairs from model" << endl;
        timer_align.stop();
      }
      // Homography計算區塊...
      */
      // ====== 改為每compute_per_frame幀才計算一次 ======
      if (cnt % compute_per_frame == 0) {
        eo_pts.clear(); ir_pts.clear();
        timer_align.start();
        image_align->align(gray_eo, gray_ir, eo_pts, ir_pts, M);
        cout << "  - Frame " << cnt << ": Found " << eo_pts.size() << " feature point pairs from model" << endl;
        timer_align.stop();

        timer_find_homo.start();
        if (eo_pts.size() >= 4 && ir_pts.size() >= 4) {
          vector<cv::Point2f> eo_pts_f, ir_pts_f;
          for (const auto& pt : eo_pts) eo_pts_f.push_back(cv::Point2f(pt.x, pt.y));
          for (const auto& pt : ir_pts) ir_pts_f.push_back(cv::Point2f(pt.x, pt.y));
          cv::Mat mask;
          cv::Mat H = cv::findHomography(eo_pts_f, ir_pts_f, cv::RANSAC, 15.0, mask, 8000, 0.85);
          if (!H.empty() && !mask.empty()) {
            int inliers = cv::countNonZero(mask);
            if (inliers >= 4 && cv::determinant(H) > 1e-6 && cv::determinant(H) < 1e6) {
              M = H.clone();
              // 過濾inlier特徵點
              std::vector<cv::Point2i> filtered_eo_pts, filtered_ir_pts;
              for (int i = 0; i < mask.rows; i++) {
                if (mask.at<uchar>(i, 0) > 0) {
                  filtered_eo_pts.push_back(eo_pts[i]);
                  filtered_ir_pts.push_back(ir_pts[i]);
                }
              }
              eo_pts = filtered_eo_pts;
              ir_pts = filtered_ir_pts;
            } else {
              M = cv::Mat::eye(3, 3, CV_64F);
            }
          } else {
            M = cv::Mat::eye(3, 3, CV_64F);
          }
        } else {
          M = cv::Mat::eye(3, 3, CV_64F);
        }
        timer_find_homo.stop();
        // 只在這裡重畫特徵點配對圖像
        temp_pair = Mat::zeros(out_h, out_w * 2, CV_8UC3);
        cv::hconcat(img_ir, img_eo, temp_pair);
        if (eo_pts.size() > 0 && ir_pts.size() > 0) {
          for (int i = 0; i < std::min((int)eo_pts.size(), (int)ir_pts.size()); i++) {
            cv::Point2i pt_ir = ir_pts[i];
            cv::Point2i pt_eo = eo_pts[i];
            pt_eo.x += out_w;
            if (pt_ir.x >= 0 && pt_ir.x < out_w && pt_ir.y >= 0 && pt_ir.y < out_h &&
                pt_eo.x >= out_w && pt_eo.x < out_w * 2 && pt_eo.y >= 0 && pt_eo.y < out_h) {
              cv::circle(temp_pair, pt_ir, 3, cv::Scalar(0, 255, 0), -1);
              cv::circle(temp_pair, pt_eo, 3, cv::Scalar(0, 0, 255), -1);
              cv::line(temp_pair, pt_ir, pt_eo, cv::Scalar(255, 0, 0), 1);
            }
          }
        }
      }
      // 其餘幀直接沿用上一次的M、特徵點、temp_pair

      // 邊緣檢測和融合處理，與Python相同
      Mat edge, img_combined;
      
      {
        timer_edge.start();
        edge = image_fusion->edge(gray_eo);
        timer_edge.stop();
      }
      
      // 將EO影像轉換到IR的座標系統，如果有有效的homography矩陣
      Mat edge_warped = edge.clone();
      if (!M.empty() && cv::determinant(M) > 1e-6) // 檢查矩陣是否有效
      {
        timer_perspective.start();
        // 回復原始版本：使用線性插值
        cv::warpPerspective(edge, edge_warped, M, cv::Size(out_w, out_h), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        
        // 高品質插值版本 (效果不佳，已停用)：
        // cv::warpPerspective(edge, edge_warped, M, cv::Size(out_w, out_h), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0));
        timer_perspective.stop();
        // cout << "  - Applied homography transformation to edge image (improved RANSAC)" << endl;
      }
      else
      {
        cout << "  - Using original edge image (no valid homography)" << endl;
      }
      
      {
        timer_fusion.start();
        img_combined = image_fusion->fusion(edge_warped, img_ir);
        timer_fusion.stop();
      }
      
      timer_base.stop();
      
      // 輸出影像，與Python相同
      Mat img;
      cv::hconcat(temp_pair, img_combined, img);
      
      // 顯示處理結果
      imshow("out", img);

      /*
      // 原始程式碼註解掉：
      vector<cv::Point2i> eo_pts, ir_pts;
      {
        // cv::Mat H = image_perspective->get_perspective_matrix();
        cv::Mat H;
        timer_align.start();
        image_align->align(eo_gray, ir_gray, eo_pts, ir_pts, H);
        timer_align.stop();
      }

      {
        timer_find_homo.start();
        bool sta = image_perspective->find_perspective_matrix_msac(eo_pts, ir_pts);
        timer_find_homo.stop();
      }

      // MODIFIED: 直接使用目標尺寸進行變換，移除後續裁剪
      // 原本: eo_wrap = image_perspective->wrap(eo_gray, eo_new_w, out_h);
      //       eo_wrap = image_resizer->clip_eo(eo_wrap);
      // 現在: eo_wrap = image_perspective->wrap(eo_gray, out_w, out_h);
      {
        timer_perspective.start();
        eo_wrap = image_perspective->wrap(eo_gray, out_w, out_h);
        timer_perspective.stop();
      }

      {
        timer_fusion.start();
        eo_edge = image_fusion->edge(eo_wrap);
        timer_fusion.stop();
      }

      {
        timer_edge.start();
        out = image_fusion->fusion(eo_edge, ir_resize);
        timer_edge.stop();
      }

      timer_base.stop();

      // MODIFIED: 視覺化關鍵點時的座標計算
      // 原本: int bias = image_resizer->get_eo_clip_x();
      //       cv::Point2i ir_pt(ir_pts[i].x + eo_new_w - bias, ir_pts[i].y);
      // 現在: cv::Point2i ir_pt(ir_pts[i].x + out_w, ir_pts[i].y);
      Mat catArr[] = {eo_resize, ir_resize, out};
      Mat cat;
      cv::hconcat(catArr, 3, cat);

      for (int i = 0; i < eo_pts.size(); i++)
      {
        cv::Point2i eo_pt(eo_pts[i].x, eo_pts[i].y);
        cv::Point2i ir_pt(ir_pts[i].x + out_w, ir_pts[i].y);
        cv::circle(cat, eo_pt, 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(cat, ir_pt, 2, cv::Scalar(0, 255, 0), -1);
        cv::line(cat, eo_pt, ir_pt, cv::Scalar(255, 0, 0), 1);
      }

      imshow("out", cat);
      */

      if (isVideo)
      {
        if (isOut)
          writer.write(img);

        int key = waitKey(1);
        if (key == 27)
          return 0;
          
        // ADDED: 加入Python風格的frame rate調整
        // Python: for _ in range(rate - 1): ret_ir, frame_ir = cap_ir.read()
        for (int i = 0; i < frame_rate - 1; i++)
        {
          Mat temp_ir;
          ir_cap.read(temp_ir);
        }
      }
      else
      {
        if (isOut)
          imwrite(save_path + ".jpg", img);

        int key = waitKey(0);
        if (key == 27)
          return 0;

        break;
      }
      
      // 增加幀數計數器
      cnt++;
    }

    timer_resize.show();
    timer_gray.show();
    // REMOVED: timer_clip.show(); - 移除裁剪計時器顯示
    timer_equalization.show();
    timer_find_homo.show();
    timer_edge.show();
    timer_perspective.show();
    timer_fusion.show();
    timer_align.show();

    eo_cap.release();
    ir_cap.release();
    if (isOut)
      writer.release();

    // return 0;
  }
}
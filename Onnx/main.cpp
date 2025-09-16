#include <ratio>
#include <chrono>
#include <string>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <filesystem>
#include <cmath>
#include <limits>
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

// =============== 選擇版本：註解掉不需要的版本 ===============
// 版本 1: ONNX 版本
#include "lib_image_fusion/include/core_image_align_onnx.h"
#include "lib_image_fusion/src/core_image_align_onnx.cpp"

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

  config.emplace("Vcut_x", 0);
  config.emplace("Vcut_y", 0);
  config.emplace("Vcut_h", -1); // -1 means no cut, use full image height
  config.emplace("Vcut_w", -1); // -1 means no cut, use full image width

  config.emplace("output_width", 480);
  config.emplace("output_height", 360);

  config.emplace("pred_width", 320);//480,360
  config.emplace("pred_height", 240);// 640 480

  config.emplace("fusion_shadow", true);
  config.emplace("fusion_edge_border", 2);  // 增加邊緣寬度從1到2
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

  // 平滑 homography 相關參數
  config.emplace("smooth_max_translation_diff", 15.0);  // 最大允許平移差異 (像素) - 降低閾值
  config.emplace("smooth_max_rotation_diff", 0.02);     // 最大允許旋轉差異 (弧度) - 降低閾值
  config.emplace("smooth_alpha", 0.03);                 // 平滑係數 (0-1, 越小越平滑) - 降低係數

  config.emplace("skip_frames", nlohmann::json::object());

}

cv::Mat cropImage(const cv::Mat& sourcePic, int x, int y, int w, int h) {
    // 邊界檢查，確保不超出原圖
    int crop_x = std::max(0, x);
    int crop_y = std::max(0, y);
    int crop_w = w;
    int crop_h = h;
    if (w < 0) {
        crop_w = sourcePic.cols - crop_x;
    }
    if (h < 0) {
        crop_h = sourcePic.rows - crop_y;
    }
    crop_w = std::min(crop_w, sourcePic.cols - crop_x);
    crop_h = std::min(crop_h, sourcePic.rows - crop_y);
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

// 平滑 Homography 管理器類
class SmoothHomographyManager {
private:
    double max_translation_diff;
    double max_rotation_diff;
    double smooth_alpha;
    cv::Mat previous_homo;
    
public:
    SmoothHomographyManager(double max_trans_diff = 30.0, double max_rot_diff = 0.03, double alpha = 0.05) 
        : max_translation_diff(max_trans_diff), max_rotation_diff(max_rot_diff), smooth_alpha(alpha) {}
    
    // 計算兩個 homography 矩陣的差異
    std::pair<double, double> calculateHomographyDifference(const cv::Mat& homo1, const cv::Mat& homo2) {
        if (homo1.empty() || homo2.empty()) {
            return {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
        }
        
        // 計算平移差異
        double translation_diff = sqrt(pow(homo1.at<double>(0, 2) - homo2.at<double>(0, 2), 2) +
                                     pow(homo1.at<double>(1, 2) - homo2.at<double>(1, 2), 2));
        
        // 計算旋轉差異（通過2x2左上角矩陣）
        double angle1 = atan2(homo1.at<double>(1, 0), homo1.at<double>(0, 0));
        double angle2 = atan2(homo2.at<double>(1, 0), homo2.at<double>(0, 0));
        double rotation_diff = abs(angle1 - angle2);
        
        // 處理角度循環問題
        if (rotation_diff > M_PI) {
            rotation_diff = 2 * M_PI - rotation_diff;
        }
        
        return {translation_diff, rotation_diff};
    }
    
    // 判斷是否應該更新 homography
    bool shouldUpdateHomography(const cv::Mat& new_homo) {
        if (previous_homo.empty()) {
            return true;
        }
        
        auto [trans_diff, rot_diff] = calculateHomographyDifference(previous_homo, new_homo);
        
        // 如果差異太大，不更新
        if (trans_diff > max_translation_diff || rot_diff > max_rotation_diff) {
            return false;
        }
        
        return true;
    }
    
    // 更新 homography 並進行平滑處理
    cv::Mat updateHomography(const cv::Mat& new_homo) {
        if (new_homo.empty()) {
            return previous_homo;
        }
        
        // 如果這是第一次更新，直接使用新的
        if (previous_homo.empty()) {
            previous_homo = new_homo.clone();
            return new_homo;
        }
        
        // 如果應該更新，使用平滑混合
        if (shouldUpdateHomography(new_homo)) {
            // 平滑混合: smooth_alpha * 新的 + (1-smooth_alpha) * 舊的
            cv::Mat smoothed_homo = smooth_alpha * new_homo + (1 - smooth_alpha) * previous_homo;
            previous_homo = smoothed_homo.clone();
            return smoothed_homo;
        } else {
            // 差異太大，保持前一次的 homography
            return previous_homo;
        }
    }
    
    // 獲取當前 homography
    cv::Mat getCurrentHomography() {
        return previous_homo;
    }
    
    // 設置參數
    void setParameters(double max_trans_diff, double max_rot_diff, double alpha) {
        max_translation_diff = max_trans_diff;
        max_rotation_diff = max_rot_diff;
        smooth_alpha = alpha;
    }
};

// GT homography 讀取函數
cv::Mat readGTHomography(const std::string& gt_path, const std::string& img_name) {
  std::string json_file = gt_path + "/IR_" + img_name + ".json";
  
  if (!std::filesystem::exists(json_file)) {
    std::cout << "GT file not found: " << json_file << std::endl;
    return cv::Mat();
  }
  
  try {
    std::ifstream file(json_file);
    nlohmann::json j;
    file >> j;
    
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    auto h_array = j["H"];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        H.at<double>(i, j) = h_array[i][j];
      }
    }
    std::cout << "GT homography loaded from: " << json_file << std::endl;
    return H;
  } catch (const std::exception& e) {
    std::cout << "Error reading GT homography from " << json_file << ": " << e.what() << std::endl;
    return cv::Mat();
  }
}

// 計算 homography 誤差函數
// 計算特徵點MSE誤差函數 - 與LibTorch版本完全一致
double calcFeaturePointMSE(const cv::Mat& homo_pred, const cv::Mat& homo_gt, 
                          const std::vector<cv::Point2i>& eo_pts) {
    if (homo_pred.empty() || homo_gt.empty() || eo_pts.empty()) return -1.0;
    
    // 將EO特徵點轉換為float格式
    std::vector<cv::Point2f> eo_pts_f;
    for (const auto& pt : eo_pts) {
        eo_pts_f.push_back(cv::Point2f(pt.x, pt.y));
    }
    
    // 1. eo的點 * eo_homo_pred(矩陣) = kpts_pred(轉換後的點)
    std::vector<cv::Point2f> kpts_pred;
    cv::perspectiveTransform(eo_pts_f, kpts_pred, homo_pred);
    
    // 2. eo的點 * homo_gt(自己建立的gt) = kpts_gt
    std::vector<cv::Point2f> kpts_gt;
    cv::perspectiveTransform(eo_pts_f, kpts_gt, homo_gt);
    
    // 3. kpts_pred & kpts_gt計算所有特徵點計算距離差(MSE)
    double total_squared_error = 0.0;
    int valid_points = 0;
    
    for (size_t i = 0; i < kpts_pred.size() && i < kpts_gt.size(); ++i) {
        double dx = kpts_pred[i].x - kpts_gt[i].x;
        double dy = kpts_pred[i].y - kpts_gt[i].y;
        double squared_distance = dx * dx + dy * dy;
        total_squared_error += squared_distance;
        valid_points++;
    }
    
    if (valid_points == 0) return -1.0;
    
    // 返回MSE (Mean Squared Error)
    return total_squared_error / valid_points;
}

int main(int argc, char **argv)
{
  // ===== 移除隨機種子設定以與LibTorch保持一致 =====
  // cv::setRNGSeed(12345);  // 註釋掉以與LibTorch保持一致
  
  // 新增: 追蹤特徵點座標範圍
  int min_x = INT_MAX, max_x = INT_MIN;
  int min_y = INT_MAX, max_y = INT_MIN;
  string current_video = "";

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

  // get Vcut parameter
  bool isVideoCut = config["VideoCut"];
  int Vcut_x = config["Vcut_x"];//3840*2160
  int Vcut_y = config["Vcut_y"];
  int Vcut_w = config["Vcut_w"];
  int Vcut_h = config["Vcut_h"];

  bool isPictureCut = config["PictureCut"];
  int Pcut_x = config["Pcut_x"];//1920*1080
  int Pcut_y = config["Pcut_y"];
  int Pcut_w = config["Pcut_w"];
  int Pcut_h = config["Pcut_h"];


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
  // 新增：插值方式
  // get perspective parameter
  bool perspective_check = config["perspective_check"];
  float perspective_distance = config["perspective_distance"];
  float perspective_accuracy = config["perspective_accuracy"];

  // get align parameter
  float align_angle_mean = config["align_angle_mean"];
  float align_angle_sort = config["align_angle_sort"];
  float align_distance_last = config["align_distance_last"];
  float align_distance_line = config["align_distance_line"];

  // get smooth homography parameter
  double smooth_max_translation_diff = config["smooth_max_translation_diff"];
  double smooth_max_rotation_diff = config["smooth_max_rotation_diff"];
  double smooth_alpha = config["smooth_alpha"];

  // GT homography 路徑
  std::string gt_homo_base_path = "/circ330/HomoLabels320240/Version3";

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
    cout << "\tSmooth Max Translation Diff: " << smooth_max_translation_diff << endl;
    cout << "\tSmooth Max Rotation Diff: " << smooth_max_rotation_diff << endl;
    cout << "\tSmooth Alpha: " << smooth_alpha << endl;
  }

  // ----- 創建共用的非模型實例（模型將在每輪重新初始化） -----
  std::cout << "\n=== Initializing shared non-model instances ===" << std::endl;
  
  // Create shared instances that will be used for all images (除了模型)
  auto image_gray = core::ImageToGray::create_instance(core::ImageToGray::Param());

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

  // =============== 注意：模型實例將在每輪重新創建 ===============
  std::cout << "Non-model instances initialized. Model will be recreated each round." << std::endl;














  // ----- Start Processing All Files -----
  // ===== 臨時測試功能：運行1輪所有圖片來測試一致性（之後可以刪除） =====
  const int test_rounds = 300; // 測試輪數（之後可以刪除這個變數）
  




  for (int round = 1; round <= test_rounds; round++) {
    std::cout << "\n=== Processing Round " << round << "/" << test_rounds << " ===" << std::endl;
    
    // =============== 每輪重新初始化模型實例（清除、重新讀取、warm-up） ===============
    std::cout << "Round " << round << ": Creating new ImageAlign model instance..." << std::endl;
    auto image_align = core::ImageAlignONNX::create_instance(
        core::ImageAlignONNX::Param()
            .set_size(pred_w, pred_h, out_w, out_h)
            .set_model(device, model_path, pred_mode)
            .set_bias(0, 0));
    
    std::cout << "Round " << round << ": Model initialization and warm-up completed." << std::endl;
    
    // 在每輪開始時在CSV中加入分隔符（之後可以刪除這個功能）
    std::ofstream csv_separator("image_feature_mse_errors.csv", std::ios::app);
    if (csv_separator.is_open()) {
      csv_separator << "-------------------------,Round_" << round << ",-------------------------,-------------------------" << std::endl;
      csv_separator.close();
    }
    
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
    VideoWriter writer_fusion; // 新增：只輸出融合圖的影片
    if (isVideo)
    {
      eo_cap.open(eo_path);
      ir_cap.open(ir_path);
      skip_frames(eo_path, eo_cap, config);
      skip_frames(ir_path, ir_cap, config);

      eo_w = (int)eo_cap.get(3), eo_h = (int)eo_cap.get(4);
      ir_w = (int)ir_cap.get(3), ir_h = (int)ir_cap.get(4);
      
      int fps_ir = (int)ir_cap.get(cv::CAP_PROP_FPS);
      int fps_eo = (int)eo_cap.get(cv::CAP_PROP_FPS);
      frame_rate = fps_ir / fps_eo;
      
      cout << "  - IR: " << fps_ir << " fps, " << ir_w << "x" << ir_h << endl;
      cout << "  - EO: " << fps_eo << " fps, " << eo_w << "x" << eo_h << endl;
      cout << "  - Rate: " << frame_rate << " (IR:EO)" << endl;
      
      if (isOut)
      {
        writer.open(save_path + "_shadow.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps_ir, cv::Size(out_w * 3, out_h));
        // writer_fusion.open(save_path + "_fusion.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps_ir, cv::Size(out_w, out_h));
      }
    }
    else
    {
      Mat eo = imread(eo_path);
      Mat ir = imread(ir_path);
      eo_w = eo.cols, eo_h = eo.rows;
      ir_w = ir.cols, ir_h = ir.rows;
    }

    // =============== 使用已經初始化的共用模型實例 ===============

    // 開始計時
    auto timer_base = core::Timer("All");
    auto timer_resize = core::Timer("Resize");
    auto timer_gray = core::Timer("Gray");
    auto timer_equalization = core::Timer("Equalization");
    auto timer_perspective = core::Timer("Perspective");
    auto timer_find_homo = core::Timer("Homo");
    auto timer_fusion = core::Timer("Fusion");
    auto timer_edge = core::Timer("Edge");
    auto timer_align = core::Timer("Align");

    // 讀取影片
    Mat eo, ir;
    
    int cnt = 0;  // 幀數計數器
    cv::Mat M;    // Homography矩陣
    Mat temp_pair = Mat::zeros(out_h, out_w * 2, CV_8UC3);  // 儲存特徵點配對圖像
    std::vector<cv::Point2i> eo_pts, ir_pts; // 保留特徵點
    const int compute_per_frame = 50; // 每50幀做一次
    
    // 初始化平滑 homography 管理器
    SmoothHomographyManager homo_manager(smooth_max_translation_diff, smooth_max_rotation_diff, smooth_alpha);
    int fallback_count = 0; // 新增：連續 fallback 次數計數器

    while (1)
    {
      if (isVideo)
      {
        ir_cap.read(ir);
        eo_cap.read(eo);
        // 新增：eo每一幀都經過裁切（預設裁切全圖）
        if (isVideoCut) {
          eo = cropImage(eo, Vcut_x, Vcut_y, Vcut_w, Vcut_h);
          //3840*2160
        }
      }
      else
      {
        eo = cv::imread(eo_path);
        ir = cv::imread(ir_path);
        
        // 提取圖片名稱用於CSV記錄
        std::string img_name = eo_path.substr(eo_path.find_last_of("/\\") + 1);
        size_t dot_pos = img_name.find_last_of(".");
        if (dot_pos != std::string::npos) {
          img_name = img_name.substr(0, dot_pos);
        }
        size_t eo_pos = img_name.find("_EO");
        if (eo_pos != std::string::npos) {
          img_name = img_name.substr(0, eo_pos);
        }
        // 設置當前圖片名稱到模型中
        image_align->set_current_image_name(img_name);
        
        // 圖片裁剪
        // 1. 讀取eo,ir (已完成)
        // 2. eo裁切並resize320240,ir resize320240
        if (isPictureCut) {
          eo = cropImage(eo, Pcut_x, Pcut_y, Pcut_w, Pcut_h);
        }
        // resize到320x240
        cv::Mat eo_final, ir_final;
        // cv::resize(eo, eo_final, cv::Size(320, 240), 0, 0, cv::INTER_CUBIC);
        // cv::resize(ir, ir_final, cv::Size(320, 240), 0, 0, cv::INTER_CUBIC);
        cv::resize(eo, eo_final, cv::Size(320, 240), 0, 0, cv::INTER_AREA);
        cv::resize(ir, ir_final, cv::Size(320, 240), 0, 0, cv::INTER_AREA);
        
        
        // 3. 兩張圖片經過gray並擴增到3channel
        cv::Mat gray_eo, gray_ir;
        cv::cvtColor(eo_final, gray_eo, cv::COLOR_BGR2GRAY);
        cv::cvtColor(ir_final, gray_ir, cv::COLOR_BGR2GRAY);
        
        // 擴增到3通道 (雖然模型只用單通道，但為了流程一致性)
        // cv::Mat gray_eo, gray_ir;
        // cv::cvtColor(gray_eo, gray_eo, cv::COLOR_GRAY2BGR);
        // cv::cvtColor(gray_ir, gray_ir, cv::COLOR_GRAY2BGR);
        
        // 4. 兩張圖片給model做預測
        // 5. model預測出kps1,kps2,leng1,leng2
        eo_pts.clear(); ir_pts.clear();
        cv::Mat M_single;
        
        image_align->align(gray_eo, gray_ir, eo_pts, ir_pts, M_single);
        
        // 6. kps1和kps2經過ransac(8.0 ,800,0.98)之後得到縮減後的kps1,kps2
        // 7. kps1和kps2經過homo轉換得到一個H
        cv::Mat refined_H = M_single.clone();
        if (eo_pts.size() >= 4 && ir_pts.size() >= 4) {
          std::vector<cv::Point2f> eo_pts_f, ir_pts_f;
          for (const auto& pt : eo_pts) eo_pts_f.push_back(cv::Point2f(pt.x, pt.y));
          for (const auto& pt : ir_pts) ir_pts_f.push_back(cv::Point2f(pt.x, pt.y));
          cv::Mat mask;
          cv::Mat H = cv::findHomography(eo_pts_f, ir_pts_f, cv::RANSAC, 8.0, mask, 1000, 0.99);
          if (!H.empty() && !mask.empty()) {
            int inliers = cv::countNonZero(mask);
            if (inliers >= 4 && cv::determinant(H) > 1e-6 && cv::determinant(H) < 1e6) {
              refined_H = H;
              // 先過濾 inlier 特徵點
              std::vector<cv::Point2i> filtered_eo_pts, filtered_ir_pts;
              for (int i = 0; i < mask.rows; i++) {
                if (mask.at<uchar>(i, 0) > 0) {
                  filtered_eo_pts.push_back(eo_pts[i]);
                  filtered_ir_pts.push_back(ir_pts[i]);
                }
              }
              eo_pts = filtered_eo_pts;
              ir_pts = filtered_ir_pts;
              
              // // 使用過濾後的特徵點重新計算 homography (最小二乘法)
              // if (eo_pts.size() >= 4) {
              //   std::vector<cv::Point2f> refined_eo_pts_f, refined_ir_pts_f;
              //   for (const auto& pt : eo_pts) refined_eo_pts_f.push_back(cv::Point2f(pt.x, pt.y));
              //   for (const auto& pt : ir_pts) refined_ir_pts_f.push_back(cv::Point2f(pt.x, pt.y));
                
              //   // 使用最小二乘法重新計算 homography (特徵點已過濾，不需要再用RANSAC)
              //   cv::Mat H_refined = cv::findHomography(refined_eo_pts_f, refined_ir_pts_f, 0); // 0 = 最小二乘法
              //   if (!H_refined.empty() && cv::determinant(H_refined) > 1e-6 && cv::determinant(H_refined) < 1e6) {
              //     refined_H = H_refined;
              //     std::cout << "    -> 使用 " << eo_pts.size() << " 個過濾後特徵點重新計算 homography" << std::endl;
              //   } else {
              //     refined_H = H; // 重新計算失敗，使用原始RANSAC結果
              //     std::cout << "    -> 重新計算失敗，使用原始RANSAC結果" << std::endl;
              //   }
              // } else {
              //   refined_H = H; // 特徵點不足，使用原始RANSAC結果
              //   std::cout << "    -> 過濾後特徵點不足，使用原始RANSAC結果" << std::endl;
              // }
            }
          }
        }
        // 使用 refined homography
        M = refined_H.empty() ? cv::Mat::eye(3, 3, CV_64F) : refined_H.clone();
        
        // 使用final_output_size進行後續處理
        // cv::Mat eo_final, ir_final;
        // cv::resize(eo_resized, eo_final, cv::Size(out_w, out_h));
        // cv::resize(ir_resized, ir_final, cv::Size(out_w, out_h));
        
        // ========== 圖片模式下組合顯示 ==========
        // 準備 temp_pair：左邊IR，右邊EO經過homo變換
        cv::Mat temp_pair = cv::Mat::zeros(out_h, out_w * 2, CV_8UC3);
        ir_final.copyTo(temp_pair(cv::Rect(0, 0, out_w, out_h)));
        
        // EO經過homography變換
        cv::Mat eo_warped;
        if (!M.empty() && cv::determinant(M) > 1e-6) {
          cv::warpPerspective(eo_final, eo_warped, M, cv::Size(out_w, out_h));
        } else {
          eo_warped = eo_final.clone();
        }
        eo_warped.copyTo(temp_pair(cv::Rect(out_w, 0, out_w, out_h)));
        
        // 邊緣檢測
        cv::Mat edge = image_fusion->edge(gray_eo);
        // warp edge
        cv::Mat edge_warped = edge.clone();
        if (!M.empty() && cv::determinant(M) > 1e-6) {
          cv::warpPerspective(edge, edge_warped, M, cv::Size(out_w, out_h));
        }
        // 融合
        cv::Mat img_combined = image_fusion->fusion(edge_warped, ir_final);
        // 組合顯示
        cv::Mat img = cv::Mat(out_h, out_w * 3, CV_8UC3);
        temp_pair.copyTo(img(cv::Rect(0, 0, out_w * 2, out_h)));
        img_combined.copyTo(img(cv::Rect(out_w * 2, 0, out_w, out_h)));
        // 顯示 - 改為保存圖片而不是顯示
        // imshow("out", img); // 註解掉以避免GUI錯誤
        std::cout << "Generated fusion result: " << out_w * 3 << "x" << out_h << std::endl;
        if (isOut) {
          imwrite(save_path + ".jpg", img);//圖片輸出
          std::cout << "Saved fusion result to: " << save_path + ".jpg" << std::endl;
        }
        
        // 8. kps1使用H得到eo_pred_H,kps1使用gt_H得到eo_gt_H，計算兩個所有的特徵點用相同位置對應對方的一個特徵點並計算mse誤差
        std::cout << "\n=== Computing Feature Point MSE Error ===" << std::endl;
        std::cout << "EO Path: " << eo_path << std::endl;
        std::cout << "IR Path: " << ir_path << std::endl;
        
        // 讀取 GT homography（使用之前提取的 img_name）
        cv::Mat gt_homo = readGTHomography(gt_homo_base_path, img_name);
        
        if (!gt_homo.empty() && !eo_pts.empty()) {
          // 直接使用已經計算出的 homography M
          cv::Mat final_M = M.empty() ? cv::Mat::eye(3, 3, CV_64F) : M;
          
          // 計算特徵點MSE誤差 (與LibTorch版本完全一致)
          double feature_mse_error = calcFeaturePointMSE(final_M, gt_homo, eo_pts);
          
          // 寫入 CSV
          std::string csv_filename = "image_feature_mse_errors.csv";
          std::ofstream csv_file;
          bool file_exists = std::filesystem::exists(csv_filename);
          csv_file.open(csv_filename, std::ios::app);
          
          if (!file_exists) {
            csv_file << "Image_Name,Image_Size,Feature_Points,MSE_Error\n";
          }
          
          std::string size_str = "320*240";  // 使用模型預測尺寸
          csv_file << img_name << "," << size_str << "," << eo_pts.size() << "," << feature_mse_error << "\n";
          csv_file.close();
          
          std::cout << "    Feature Point MSE Error: " << feature_mse_error << " px^2" << std::endl;
          std::cout << "    Used " << eo_pts.size() << " feature points" << std::endl;
          std::cout << "CSV result saved to image_feature_mse_errors.csv" << std::endl;
        } else {
          if (gt_homo.empty()) {
            std::cout << "GT homography not found for image: " << img_name << std::endl;
          }
          if (eo_pts.empty()) {
            std::cout << "No feature points found for MSE calculation" << std::endl;
          }
        }
        
        // int key = waitKey(0); // 註解掉以避免GUI錯誤
        // if (key == 27)
        //   return 0;
        std::cout << "Processing completed for single image." << std::endl;
        // 完成後直接break，避免進入影片專用流程
        break;
      }
      // 退出迴圈條件
      if (eo.empty() || ir.empty())
        break;

      // 幀數計數
      timer_base.start();

      // 新程式碼：按照Python版本
      Mat img_ir, img_eo, gray_ir, gray_eo;
      
      // Resize圖像
      {
        timer_resize.start();
        cv::resize(ir, img_ir, cv::Size(out_w, out_h));
        cv::resize(eo, img_eo, cv::Size(out_w, out_h));
        timer_resize.stop();
      }
      
      // 轉換為灰度圖像
      {
        timer_gray.start();
        cv::cvtColor(img_ir, gray_ir, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img_eo, gray_eo, cv::COLOR_BGR2GRAY);
        timer_gray.stop();
      }
      
      // 每50幀計算一次特徵點
      if (cnt % compute_per_frame == 0) {
        eo_pts.clear(); ir_pts.clear();
        timer_align.start();
        image_align->align(gray_eo, gray_ir, eo_pts, ir_pts, M);
        cout << "  - Frame " << cnt << ": Found " << eo_pts.size() << " feature point pairs from model" << endl;
        timer_align.stop();

        // 更新特徵點座標範圍
        for (const auto& pt : eo_pts) {
          min_x = std::min(min_x, pt.x);
          max_x = std::max(max_x, pt.x);
          min_y = std::min(min_y, pt.y);
          max_y = std::max(max_y, pt.y);
        }

        timer_find_homo.start();
        if (eo_pts.size() >= 4 && ir_pts.size() >= 4) {
          vector<cv::Point2f> eo_pts_f, ir_pts_f;
          for (const auto& pt : eo_pts) eo_pts_f.push_back(cv::Point2f(pt.x, pt.y));
          for (const auto& pt : ir_pts) ir_pts_f.push_back(cv::Point2f(pt.x, pt.y));
          cv::Mat mask;
          
          // 移除隨機種子設定以與LibTorch保持一致
          // cv::setRNGSeed(12345);  // 註釋掉以與LibTorch保持一致
          
          cv::Mat H = cv::findHomography(eo_pts_f, ir_pts_f, cv::RANSAC, 8.0, mask, 800, 0.98);
          if (!H.empty() && !mask.empty()) {
            int inliers = cv::countNonZero(mask);
            if (inliers >= 4 && cv::determinant(H) > 1e-6 && cv::determinant(H) < 1e6) {
              if (homo_manager.getCurrentHomography().empty()) {
                M = homo_manager.updateHomography(H);
                fallback_count = 0; // reset
                cout << "  - Frame " << cnt << ": First homography computed" << endl;
              } else {
                std::pair<double, double> diff = homo_manager.calculateHomographyDifference(
                    homo_manager.getCurrentHomography(), H);
                double trans_diff = diff.first;
                double rot_diff = diff.second;
                cout << "  - Frame " << cnt << ": Translation diff=" << trans_diff 
                     << "px, Rotation diff=" << rot_diff << "rad" << endl;
                if (trans_diff > smooth_max_translation_diff || rot_diff > smooth_max_rotation_diff) {
                  fallback_count++;
                  cout << "    -> Difference too large, keeping previous homography (fallback_count=" << fallback_count << ")" << endl;
                  if (fallback_count >= 3) {
                    // 強制接受這次的 homography 並初始化
                    cout << "    -> Fallback >= 3, force accept and reset!" << endl;
                    homo_manager = SmoothHomographyManager(smooth_max_translation_diff, smooth_max_rotation_diff, smooth_alpha);
                    M = homo_manager.updateHomography(H); // 這次直接設為新的
                    fallback_count = 0;
                  } else {
                    M = homo_manager.getCurrentHomography();
                  }
                } else {
                  cout << "    -> Difference acceptable, smoothly updating homography (alpha=" 
                       << smooth_alpha << ")" << endl;
                  M = homo_manager.updateHomography(H);
                  fallback_count = 0;
                }
              }
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
              // 如果品質不好，使用之前的 homography
              M = homo_manager.getCurrentHomography();
              if (M.empty()) {
                M = cv::Mat::eye(3, 3, CV_64F);
              }
              cout << "  - Frame " << cnt << ": Poor quality homography, using previous" << endl;
            }
          } else {
            // 如果無法計算 homography，使用之前的
            M = homo_manager.getCurrentHomography();
            if (M.empty()) {
              M = cv::Mat::eye(3, 3, CV_64F);
            }
            cout << "  - Frame " << cnt << ": Cannot compute homography, using previous" << endl;
          }
        } else {
          // 如果特徵點不足，使用之前的 homography
          M = homo_manager.getCurrentHomography();
          if (M.empty()) {
            M = cv::Mat::eye(3, 3, CV_64F);
          }
          cout << "  - Frame " << cnt << ": Insufficient feature points, using previous" << endl;
        }
        timer_find_homo.stop();
        
        // 在resize後的圖片上建立特徵點配對圖像
        temp_pair = Mat::zeros(out_h, out_w * 2, CV_8UC3);
        img_ir.copyTo(temp_pair(cv::Rect(0, 0, out_w, out_h)));
        
        // 對EO進行homography變換
        cv::Mat eo_warped;
        if (!M.empty() && cv::determinant(M) > 1e-6) {
          cv::warpPerspective(img_eo, eo_warped, M, cv::Size(out_w, out_h));
        } else {
          eo_warped = img_eo.clone();
        }
        
        // 新增：將eo_warped放大到與ir相同尺寸（而不是out_w x out_h）
        cv::Mat eo_warped_fullsize;
        cv::resize(eo_warped, eo_warped_fullsize, cv::Size(ir.cols, ir.rows));
        
        // 將放大後的eo_warped再resize回輸出尺寸放到temp_pair中間
        cv::Mat eo_warped_resized;
        cv::resize(eo_warped_fullsize, eo_warped_resized, cv::Size(out_w, out_h));
        eo_warped_resized.copyTo(temp_pair(cv::Rect(out_w, 0, out_w, out_h)));
        
        // 參考main0712.cpp的畫點劃線方式
        if (eo_pts.size() > 0 && ir_pts.size() > 0) {
          for (int i = 0; i < std::min((int)eo_pts.size(), (int)ir_pts.size()); i++) {
            cv::Point2i pt_ir = ir_pts[i];
            cv::Point2i pt_eo = eo_pts[i];
            pt_eo.x += out_w; // EO特徵點在右側圖片，需要加上偏移
            
            // 邊界檢查，確保點在有效範圍內
            if (pt_ir.x >= 0 && pt_ir.x < out_w && pt_ir.y >= 0 && pt_ir.y < out_h &&
                pt_eo.x >= out_w && pt_eo.x < out_w * 2 && pt_eo.y >= 0 && pt_eo.y < out_h) {
              // 繪製特徵點（使用填充圓圈）
              cv::circle(temp_pair, pt_ir, 3, cv::Scalar(0, 255, 0), -1); // IR: 綠色
              cv::circle(temp_pair, pt_eo, 3, cv::Scalar(0, 0, 255), -1); // EO: 紅色
              
              // 繪製匹配線
              cv::line(temp_pair, pt_ir, pt_eo, cv::Scalar(255, 0, 0), 1); // 藍色線
            }
          }
        }
      } else {
        // 非計算幀，使用當前的 homography
        M = homo_manager.getCurrentHomography();
        if (M.empty()) {
          M = cv::Mat::eye(3, 3, CV_64F);
        }
      }

      // 邊緣檢測和融合
      Mat edge, img_combined;
      {
        timer_edge.start();
        edge = image_fusion->edge(gray_eo);
        timer_edge.stop();
      }
      // 將EO影像轉換到IR的座標系統，如果有有效的homography矩陣
      Mat edge_warped = edge.clone();
      if (!M.empty() && cv::determinant(M) > 1e-6) {
        timer_perspective.start();
        cv::warpPerspective(edge, edge_warped, M, cv::Size(out_w, out_h));
        timer_perspective.stop();
      }
      {
        timer_fusion.start();
        img_combined = image_fusion->fusion(edge_warped, img_ir);
        timer_fusion.stop();
      }
      timer_base.stop();
      // 輸出影像，確保所有影像尺寸正確
      Mat img;
      cv::Size target_size(out_w, out_h);
      if (temp_pair.size() != cv::Size(out_w * 2, out_h)) {
        cv::resize(temp_pair, temp_pair, cv::Size(out_w * 2, out_h));
      }
      if (img_combined.size() != target_size) {
        cv::resize(img_combined, img_combined, target_size);
      }
      img = cv::Mat(out_h, out_w * 3, CV_8UC3);
      temp_pair.copyTo(img(cv::Rect(0, 0, out_w * 2, out_h)));
      img_combined.copyTo(img(cv::Rect(out_w * 2, 0, out_w, out_h)));
      // 影片模式下直接寫入影片，不顯示GUI
      if (isVideo) {
        if (isOut) {
          writer.write(img);
        }
        // int key = waitKey(1); // 註解掉以避免GUI錯誤
        // if (key == 27)
        //   return 0;
        for (int i = 0; i < frame_rate - 1; i++) {
          Mat temp_ir;
          ir_cap.read(temp_ir);
        }
      } else {
        if (isOut)
          imwrite(save_path + ".jpg", img); //影片輸出
        // int key = waitKey(0); // 註解掉以避免GUI錯誤
        // if (key == 27)
        //   return 0;
        std::cout << "Frame processed and saved." << std::endl;
        break;
      }
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
    if (isOut)
      writer_fusion.release(); // 新增：釋放融合影片

    // return 0;
    
    // =============== 每輪結束時清理模型實例 ===============
    std::cout << "Round " << round << ": Cleaning up model instance for next round..." << std::endl;
    // C++ 的 smart pointer 會在作用域結束時自動清理
    // image_align 在下一次迴圈開始時會被重新創建
  }
  
  // ===== 結束測試輪數迴圈（之後可以刪除） =====
  }
  std::cout << "\n=== Completed all " << test_rounds << " test rounds ===" << std::endl;
}
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

using namespace cv;
using namespace std;

class HomographyValidator {
private:
    string output_dir;
    string log_file;
    vector<Mat> stored_matrices;
    vector<string> matrix_info;
public:
    HomographyValidator(const string& output_dir = "./output/homo_validation") 
        : output_dir(output_dir) {

        if (!filesystem::exists(output_dir)) {
            filesystem::create_directories(output_dir);
        }
        log_file = output_dir + "/validation_log.txt";
    }
    void saveHomographyResult(const Mat& homography, int frame_num, 
                             int inliers, double quality, const string& params_info) {
        if (homography.empty()) {
            cout << "Warning: Empty homography matrix for frame " << frame_num << endl;
            return;
        }

        string timestamp = getCurrentTimestamp();
        string filename = output_dir + "/homo_frame_" + to_string(frame_num) + 
                         "_" + timestamp + ".yml";

        FileStorage fs(filename, FileStorage::WRITE);
        fs << "frame_number" << frame_num;
        fs << "timestamp" << timestamp;
        fs << "homography_matrix" << homography;
        fs << "inliers_count" << inliers;
        fs << "quality_score" << quality;
        fs << "ransac_params" << params_info;
        fs << "determinant" << determinant(homography);
        fs.release();

        stored_matrices.push_back(homography.clone());
        matrix_info.push_back("Frame:" + to_string(frame_num) + " Quality:" + to_string(quality));

        writeToLog(frame_num, homography, inliers, quality, params_info);
        cout << "✓ Saved homography for frame " << frame_num << " to " << filename << endl;
    }
    double compareHomography(const Mat& H1, const Mat& H2, double tolerance = 1e-6) {
        if (H1.empty() || H2.empty()) return -1.0;

        Mat H1_norm = H1.clone();
        Mat H2_norm = H2.clone();
        if (abs(H1_norm.at<double>(2,2)) > tolerance) H1_norm /= H1_norm.at<double>(2,2);
        if (abs(H2_norm.at<double>(2,2)) > tolerance) H2_norm /= H2_norm.at<double>(2,2);

        Mat diff = H1_norm - H2_norm;
        double frobenius_norm = norm(diff, NORM_L2);

        return exp(-frobenius_norm);
    }
    vector<string> loadAndCompare(int frame_num, const Mat& current_homo, 
                                  double similarity_threshold = 0.95) {
        vector<string> comparison_results;

        string pattern = output_dir + "/homo_frame_" + to_string(frame_num) + "_*.yml";
        glob_t glob_result;
        int glob_status = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
        if (glob_status == 0) {
            for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
                string filename = glob_result.gl_pathv[i];

                FileStorage fs(filename, FileStorage::READ);
                if (!fs.isOpened()) continue;
                Mat historical_homo;
                string timestamp;
                double quality;
                fs["homography_matrix"] >> historical_homo;
                fs["timestamp"] >> timestamp;
                fs["quality_score"] >> quality;
                fs.release();

                double similarity = compareHomography(current_homo, historical_homo);
                bool is_consistent = similarity >= similarity_threshold;
                string result = "Comparison with " + timestamp + 
                               ": similarity=" + to_string(similarity) + 
                               " (" + (is_consistent ? "PASS" : "FAIL") + ")";
                comparison_results.push_back(result);
                cout << "  " << result << endl;
            }
        }
        globfree(&glob_result);
        return comparison_results;
    }
    void generateValidationReport() {
        string report_file = output_dir + "/validation_report.txt";
        ofstream report(report_file);
        report << "=== Homography Validation Report ===" << endl;
        report << "Generated at: " << getCurrentTimestamp() << endl;
        report << "Total matrices analyzed: " << stored_matrices.size() << endl << endl;
        if (stored_matrices.size() < 2) {
            report << "Insufficient data for comparison analysis." << endl;
            report.close();
            return;
        }

        vector<double> similarities;
        for (size_t i = 0; i < stored_matrices.size(); i++) {
            for (size_t j = i + 1; j < stored_matrices.size(); j++) {
                double sim = compareHomography(stored_matrices[i], stored_matrices[j]);
                similarities.push_back(sim);
            }
        }

        if (!similarities.empty()) {
            double mean_sim = accumulate(similarities.begin(), similarities.end(), 0.0) / similarities.size();
            double min_sim = *min_element(similarities.begin(), similarities.end());
            double max_sim = *max_element(similarities.begin(), similarities.end());

            double variance = 0;
            for (double sim : similarities) {
                variance += (sim - mean_sim) * (sim - mean_sim);
            }
            double std_dev = sqrt(variance / similarities.size());
            report << "Similarity Statistics:" << endl;
            report << "  Mean: " << fixed << setprecision(6) << mean_sim << endl;
            report << "  Min:  " << min_sim << endl;
            report << "  Max:  " << max_sim << endl;
            report << "  Std:  " << std_dev << endl;

            double consistency_threshold = 0.95;
            int consistent_pairs = count_if(similarities.begin(), similarities.end(), 
                                          [consistency_threshold](double sim) { return sim >= consistency_threshold; });
            double consistency_rate = (double)consistent_pairs / similarities.size();
            report << endl << "Consistency Analysis:" << endl;
            report << "  Threshold: " << consistency_threshold << endl;
            report << "  Consistent pairs: " << consistent_pairs << "/" << similarities.size() << endl;
            report << "  Consistency rate: " << (consistency_rate * 100) << "%" << endl;
            if (consistency_rate >= 0.8) {
                report << "  ✓ GOOD: High consistency detected" << endl;
            } else if (consistency_rate >= 0.5) {
                report << "  ⚠ MODERATE: Medium consistency" << endl;
            } else {
                report << "  ✗ POOR: Low consistency, investigate parameters" << endl;
            }
        }
        report.close();
        cout << "✓ Validation report saved to " << report_file << endl;
    }
    static void setRandomSeed(int seed = 42) {
        cv::theRNG().state = seed;
        srand(seed);
        cout << "✓ Random seed set to " << seed << " for reproducibility" << endl;
    }
private:
    string getCurrentTimestamp() {
        auto now = chrono::system_clock::now();
        auto time_t = chrono::system_clock::to_time_t(now);
        auto ms = chrono::duration_cast<chrono::milliseconds>(now.time_since_epoch()) % 1000;
        stringstream ss;
        ss << put_time(localtime(&time_t), "%Y%m%d_%H%M%S");
        ss << "_" << setfill('0') << setw(3) << ms.count();
        return ss.str();
    }
    void writeToLog(int frame_num, const Mat& homography, int inliers, 
                   double quality, const string& params_info) {
        ofstream log(log_file, ios::app);
        log << getCurrentTimestamp() << " Frame:" << frame_num 
            << " Inliers:" << inliers << " Quality:" << quality 
            << " Params:" << params_info << " Det:" << determinant(homography) << endl;
        log.close();
    }
};

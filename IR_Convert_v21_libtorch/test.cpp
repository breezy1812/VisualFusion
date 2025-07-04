#include <opencv2/opencv.hpp>
int main() {
    cv::Mat img = cv::Mat::zeros(300, 300, CV_8UC3);
    cv::putText(img, "Test Window", cv::Point(50, 150),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    cv::imshow("Test", img);
    cv::waitKey(0);
    return 0;
}

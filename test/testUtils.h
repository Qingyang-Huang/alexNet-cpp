#include <opencv2/core.hpp>
#include <iostream>

void printMat(const cv::Mat& mat) {
    std::cout << "Matrix size: " << mat.size() << std::endl;
    std::cout << "Matrix data:" << std::endl;
    std::cout << mat << std::endl;
}

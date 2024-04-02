#ifndef TEST_UTILS_H 
#define TEST_UTILS_H 

#include <opencv2/core.hpp>
#include <iostream>

inline void printMat(const cv::Mat& mat) {
    std::cout << "Matrix size: " << mat.size() << std::endl;
    std::cout << "Matrix data:" << std::endl;
    std::cout << mat << std::endl;
}

inline int matEqual(const cv::Mat& mat1, const cv::Mat& mat2) {
    // Check if matrix dimensions and types match
    if (mat1.size != mat2.size || mat1.type() != mat2.type()) {
        return false;
    }

    // Check if all the elements are the same
    cv::Mat diff;
    cv::compare(mat1, mat2, diff, cv::CMP_NE); // Create a mask of elements that are different
    int nonzero = cv::countNonZero(diff); // Count non-zero values in the mask
    return nonzero == 0;
}

#endif // TEST_UTILS_H
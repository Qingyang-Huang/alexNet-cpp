#ifndef DROPOUT_H
#define DROPOUT_H

#include <opencv2/core.hpp>
#include <random>

inline cv::Mat dropout(cv::Mat input, float p = 0.5) {
    CV_Assert(p >= 0 && p <= 1); 
    cv::Mat output = input.clone();

    std::random_device rd; // 随机数生成器
    std::mt19937 gen(rd()); // 以随机设备作为种子
    std::bernoulli_distribution d(1 - p); // 以1-p的概率生成true

    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            if (!d(gen)) {
                output.at<float>(i, j) = 0.0f;
            }
        }
    }

    return output;
}

#endif // DROPOUT_H
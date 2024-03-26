#include <opencv2/opencv.hpp>
#include <iostream>
#include "model/alexNet.h"

int main() {
    // 创建一个224x224x3的矩阵，元素值在[0, 1]之间
    cv::Mat randomMat(224, 224, CV_32FC3);
    cv::randu(randomMat, cv::Scalar::all(0), cv::Scalar::all(1));

    cv::Mat label = cv::Mat::zeros(1, 10, CV_32F);
    label.at<float>(0, 2) = 1.0; // 第3个元素设置为1
    label.at<float>(0, 5) = 1.0; // 第6个元素设置为1
    label.at<float>(0, 7) = 1.0; // 第8个元素设置为1

    //学习率递减0.03~0.001
    float alpha = 0.03 - 0.029*n / (trainNum - 1);  

    alexNet = new AlexNet(224, 224, 10);
    alexNet->forward(randomMat);
    alexNet->backward(label);

    return 0;
}
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

    int epoch = 1;
    int trainNum = 1;
    AlexNet* alexNet;
    alexNet = new AlexNet(224, 224, 10);
    for (int e = 0; e < epoch; e++){
        for (int n = 0; n < trainNum; n++){
            //学习率递减
            float alpha = 0.03 - 0.029*n / (trainNum - 1);  
            alexNet->forward(randomMat);
            alexNet->backward(randomMat, label);
            alexNet->updateWeight(randomMat, alpha);
        }
    }

    

    return 0;
}
#include <opencv2/opencv.hpp>
#include <iostream>
#include "model/alexNet.h"

int main() {
    Tensor<float> randomMat = Tensor<float>(3, 224, 224);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < randomMat.shape()[0]; i++){
        for (int j = 0; j < randomMat.shape()[1]; j++){
            for (int k = 0; k < randomMat.shape()[2]; k++){
                randomMat(i, j, k) = dist(gen);
            }
        }
    }
    

    Tensor<float> label = Tensor<float>(1, 10);
    label(0, 2) = 1.0; // 第3个元素设置为1
    label(0, 5) = 1.0; // 第6个元素设置为1
    label(0, 7) = 1.0; // 第8个元素设置为1

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
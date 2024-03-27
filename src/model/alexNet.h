#ifndef ALEXNET_H
#define ALEXNET_H

#include "layers/convolutionalLayer.h"
#include "layers/poolingLayer.h"
#include "layers/outputLayer.h"

class AlexNet {
public:
    AlexNet(int inputSize_h, int inputSize_w, int outputSize);
    ~AlexNet();

private:
    int inputSize_h, inputSize_w, outputSize;
    ConvolutionalLayer* C1;
    PoolingLayer* S2;
    ConvolutionalLayer* C3;
    PoolingLayer* S4;
    ConvolutionalLayer* C5;
    ConvolutionalLayer* C6;
    ConvolutionalLayer* C7;
    PoolingLayer* S8;
    OutputLayer* O9;

    
    cv::Mat  e;   // 训练误差
    cv::Mat  L;   // 瞬时误差能量 

public:
    void forward(cv::Mat input);
    void backward(cv::Mat input, cv::Mat label);
    void updateWeight(cv::Mat input, float learningRate);
    void zeroGrad();
    void train(cv::Mat input, cv::Mat label, float learningRate);
};


#endif // ALEXNET_H
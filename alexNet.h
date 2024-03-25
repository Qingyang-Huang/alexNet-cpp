#ifndef ALEXNET_H
#define ALEXNET_H

#include "convolutionalLayer.h"
#include "poolingLayer.h"
#include "outputLayer.h"

class AlexNet {
public:
    AlexNet(int inputSize_h, int inputSize_w, int outputSize);
    ~AlexNet();

private:
    int layerNum;
    ConvolutionalLayer C1;
    PoolingLayer S2;
    ConvolutionalLayer C3;
    PoolingLayer S4;
    OutputLayer O5;
    
    Mat e;   // 训练误差
    Mat L;   // 瞬时误差能量 
}

public:
    void forward(Mat input);


#endif // ALEXNET_H
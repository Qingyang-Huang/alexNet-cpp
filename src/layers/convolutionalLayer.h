#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include <vector>
#include <opencv2/core.hpp>
#include "activation.h"

#define FULL 0
#define VALID 1
#define SAME 2


class ConvolutionalLayer {
public:
    // 构造函数
    ConvolutionalLayer(int inputWidth, int inputHeight, int kernelHeight, int kernelWidth, int padding, int stride_h, int stride_w,
                       int inChannels, int outChannels);
    // 析构函数
    ~ConvolutionalLayer();
    //getter function 
    int getInputWidth() const { return inputWidth; }

    int getInputHeight() const { return inputHeight; }

    int getKernelHeight() const { return kernelHeight; }

    int getKernelWidth() const { return kernelWidth; }

    int getStride_h() const { return stride_h; }

    int getStride_w() const { return stride_w; }

    int getPadding() const { return padding; }

    int getInChannels() const { return inChannels; }

    int getOutChannels() const { return outChannels; }

    const cv::Mat& getBias() const { return bias; }

    const cv::Mat& getV() const { return v; }

    const cv::Mat& getY() const { return y; }

    const cv::Mat& getD() const { return d; }

    const cv::Mat& getDx() const { return dx; }

    int getOutputWidth() const { return outputWidth; }

    int getOutputHeight() const { return outputHeight; }

private:
    int inputWidth;   // 输入图像的宽
    int inputHeight;  // 输入图像的长

    int outputWidth, outputHeight;

    int kernelHeight;      // 卷积核的尺寸
    int kernelWidth;
    int padding; 
    int stride_h, stride_w;

    int inChannels;   // 输入图像的数目
    int outChannels;  // 输出图像的数目

    std::vector<cv::Mat> kernel; // 四维float数组，卷积核本身是二维数据，m*n卷积核就是四维数组

    cv::Mat bias; // 偏置，个数为outChannels， 一维float数组


    cv::Mat v; // 进入激活函数的输入值,三维数组float型
    cv::Mat y; // 激活函数后神经元的输出，三维数组float型
    cv::Mat d; // 网络的局部梯度,三维数组float型

    cv::Mat dx; 

public:
    void forward(const cv::Mat inputData);
    cv::Mat conv2D(const cv::Mat& inputData, cv::Mat& kernel);
    void backward(const cv::Mat& d0);
    cv::Mat transConv2D(const cv::Mat& input, const cv::Mat& kernel);
    void updateWeight(const cv::Mat& input, float learningRate);
    void zeroGrad();
};

#endif // CONVOLUTIONALLAYER_H
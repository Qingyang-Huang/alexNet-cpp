#ifndef CONVOLUTIONAL2DLAYER_H
#define CONVOLUTIONAL2DLAYER_H

#include <vector>
#include "common/tensor.h"
#include "activation.h"


class ConvolutionalLayer2D {
public:
    // 构造函数
    ConvolutionalLayer2D(int inputHeight, int inputWidth, int kernelHeight, int kernelWidth, int stride_h, int stride_w,
                       int pad_h, int pad_w, int inChannels, int outChannels, bool useBias = false);
    // 析构函数 
    ~ConvolutionalLayer2D();
    //getter function 
    int getInputWidth() const { return inputWidth; }

    int getInputHeight() const { return inputHeight; }

    int getKernelHeight() const { return kernelHeight; }

    int getKernelWidth() const { return kernelWidth; }

    int getStride_h() const { return stride_h; }

    int getStride_w() const { return stride_w; }

    int getPad_h() const { return pad_h; }

    int getPad_w() const { return pad_w; }

    int getInChannels() const { return inChannels; }

    int getOutChannels() const { return outChannels; }

    const Tensor<float>& getKernels() const { return kernels; }

    const Tensor<float>& getBias() const { return bias; }

    const Tensor<float>& getZ() const { return z; }

    const Tensor<float>& getA() const { return a; }

    const Tensor<float>& getD() const { return d; }

    const Tensor<float>& getDx() const { return dx; }

    int getOutputWidth() const { return outputWidth; }

    int getOutputHeight() const { return outputHeight; }

private:
    int inputWidth;   // 输入图像的宽
    int inputHeight;  // 输入图像的长

    int outputWidth, outputHeight;

    int kernelHeight;      // 卷积核的尺寸
    int kernelWidth;
    int pad_w;
    int pad_h; 
    int stride_h, stride_w;

    int inChannels;   // 输入图像的数目
    int outChannels;  // 输出图像的数目

    Tensor<float> kernels; // 四维float数组，卷积核本身是二维数据，m*n卷积核就是四维数组

    bool useBias;
    Tensor<float> z;
    Tensor<float> a; 
    Tensor<float> d; // 梯度
    Tensor<float> dx; // 输入梯度
    Tensor<float> bias;

public:
    void forward(const Tensor<float>& inputData);
    void conv2D(const Tensor<float>& inputData);
    void backward(const Tensor<float>& d0);
    void transConv2D(const Tensor<float>& d0);
    Tensor<float> col2img(const Tensor<float>& colData);
    Tensor<float> img2col(const Tensor<float>& inputData);
    void updateWeight(const Tensor<float>& inputData, float learningRate);
    void zeroGrad();
    void setKernels(const Tensor<float>& kernel);
    void setBias(const Tensor<float>& bias);
};

#endif // CONVOLUTIONAL2DLAYER_H
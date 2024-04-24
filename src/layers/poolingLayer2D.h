#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H
#include <vector>
#include "common/tensor.h"
#include <cfloat> 


#define AVGPOOL 0
#define MAXPOOL 1


class PoolingLayer2D
{
public:
    PoolingLayer2D(int inputWidth, int inputHeight, int kernelHeight, int kernelWidth, int stride_h, int stride_w, int pad, int inChannels, int outChannels, int poolType = MAXPOOL);

    ~PoolingLayer2D();

    int getInputWidth() const { return inputWidth; }

    int getInputHeight() const { return inputHeight; }

    int getOutputWidth() const { return outputWidth; }

    int getOutputHeight() const { return outputHeight; }

    int getKernelHeight() const { return kernelHeight; }

    int getKernelWidth() const { return kernelWidth; }

    int getStride_h() const { return stride_h; }

    int getStride_w() const { return stride_w; }

    int getPadding() const { return padding; }

    int getInChannels() const { return inChannels; }

    int getOutChannels() const { return outChannels; }

    int getPoolType() const { return poolType; }

    const Tensor<float>& getZ() const { return z; }

    const Tensor<float>& getDx() const { return dx; }

    const Tensor<float>& getMaxPosition() const { return max_position; }  

private:
    int inputWidth;   //输入图像的宽
    int inputHeight;  //输入图像的长
    int kernelHeight; // 池化核的尺寸
    int kernelWidth;
    int stride_h, stride_w;     // 池化操作的步长
    int padding;    // 边界填充的大小
    int outputWidth, outputHeight;

    int inChannels;   //输入图像的数目
    int outChannels;  //输出图像的数目

    int poolType;     //池化的方法
    
    Tensor<float> z;   //采样函数后神经元的输出,无激活函数，三维数组float型
    Tensor<float> dx;   //网络的局部梯度,三维数组float型
    Tensor<float> max_position;   // 最大值模式下最大值的位置，三维数组float型


public:
    void forward(const Tensor<float> &inputData);

    void backward(const Tensor<float>& d0);

    void zeroGrad();

};

#endif // POOLINGLAYER_H
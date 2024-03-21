#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H
#include <vector>
#include <opencv2/core.hpp>


#define AVGPOOL 0
#define MAXPOOL 1

class PoolingLayer
{
public:
    PoolingLayer(int inputWidth, int inputHeight, int kernelSize, int stride, int pad, int inChannels, int outChannels, int poolType = MAXPOOL);

    ~PoolingLayer();

    int getInputWidth() const { return inputWidth; }

    int getInputHeight() const { return inputHeight; }

    int getOutputWidth() const { return outputWidth; }

    int getOutputHeight() const { return outputHeight; }

    int getKernelSize() const { return kernelSize; }

    int getStride() const { return stride; }

    int getPad() const { return pad; }

    int getInChannels() const { return inChannels; }

    int getOutChannels() const { return outChannels; }

    int getPoolType() const { return poolType; }

    const Mat& getBias() const { return bias; }

    const vector<Mat>& getY() const { return y; }

    const vector<Mat>& getD() const { return d; }

    const vector<Mat>& getMaxPosition() const { return max_position; }

private:
    int inputWidth;   //输入图像的宽
    int inputHeight;  //输入图像的长
    int kernelSize; // 池化核的尺寸
    int stride;     // 池化操作的步长
    int padding;    // 边界填充的大小
    int outputWidth, outputHeight;

    int inChannels;   //输入图像的数目
    int outChannels;  //输出图像的数目

    int poolType;     //池化的方法
    
    Mat bias;    //偏置, 一维float数组

    Mat y;   //采样函数后神经元的输出,无激活函数，三维数组float型
    Mat d;   //网络的局部梯度,三维数组float型
    Mat max_position;   // 最大值模式下最大值的位置，三维数组float型


public:
    void forward(cv::Mat &inputData);

    void backward(const cv::Mat& d0);

};

#endif // POOLINGLAYER_H
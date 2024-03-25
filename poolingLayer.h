#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H
#include <vector>
#include <opencv2/core.hpp>


#define AVGPOOL 0
#define MAXPOOL 1

class PoolingLayer
{
public:
    PoolingLayer(int inputWidth, int inputHeight, int kernelHeight, int kernelWidth, int stride_h, int stride_w, int pad, int inChannels, int outChannels, int poolType = MAXPOOL);

    ~PoolingLayer();

    int getInputWidth() const { return inputWidth; }

    int getInputHeight() const { return inputHeight; }

    int getOutputWidth() const { return outputWidth; }

    int getOutputHeight() const { return outputHeight; }

    int getKernelHeight() const { return kernelHeight; }

    int getStride() const { return stride; }

    int getPad() const { return pad; }

    int getInChannels() const { return inChannels; }

    int getOutChannels() const { return outChannels; }

    int getPoolType() const { return poolType; }

    const cv::Mat& getBias() const { return bias; }

    const vector<cv::Mat>& getY() const { return y; }

    const vector<cv::Mat>& getD() const { return d; }

    const vector<cv::Mat>& getDx() const { return dx; }

    const vector<cv::Mat>& getMaxPosition() const { return max_position; }

private:
    int inputWidth;   //输入图像的宽
    int inputHeight;  //输入图像的长
    int kernelHeight; // 池化核的尺寸
    int kernelWidth;
    int stride_h, int stride_w;     // 池化操作的步长
    int padding;    // 边界填充的大小
    int outputWidth, outputHeight;

    int inChannels;   //输入图像的数目
    int outChannels;  //输出图像的数目

    int poolType;     //池化的方法
    
    cv::Mat bias;    //偏置, 一维float数组

    cv::Mat y;   //采样函数后神经元的输出,无激活函数，三维数组float型
    cv::Mat dx;   //网络的局部梯度,三维数组float型
    cv::Mat max_position;   // 最大值模式下最大值的位置，三维数组float型


public:
    void forward(cv::Mat &inputData);

    void backward(const cv::Mat& d0);

};

#endif // POOLINGLAYER_H
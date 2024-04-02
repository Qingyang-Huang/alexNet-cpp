#include "poolingLayer2D.h"
#include <iostream>

PoolingLayer2D::PoolingLayer2D(int inputWidth, int inputHeight, int kernelHeight, int kernelWidth, int stride_h, int stride_w, int padding, int inChannels, int outChannels, int poolType)
  :inputWidth(inputWidth), inputHeight(inputHeight), kernelHeight(kernelHeight), kernelWidth(kernelWidth), stride_h(stride_h), stride_w(stride_w), padding(padding), inChannels(inChannels), outChannels(outChannels), poolType(poolType){
    

    outputWidth = ((inputWidth - kernelWidth + 2 * padding) / stride_w) + 1;
    outputHeight = ((inputHeight - kernelHeight + 2 * padding) / stride_h) + 1;

    y = cv::Mat::zeros(outChannels, outputHeight * outputWidth, CV_32F);
  
    max_position = cv::Mat::zeros(outChannels, outputHeight * outputWidth, CV_32SC2); // 存储位置(x, y)

    dx = cv::Mat::zeros(inChannels, inputHeight * inputWidth, CV_32F); 

}

PoolingLayer2D::~PoolingLayer2D(){
  // delete y;
  // delete max_position;
  // delete dx;
}

//forwad section 
void PoolingLayer2D::forward(const cv::Mat &inputData)
{
    int paddedHeight = inputHeight + 2 * padding;
    int paddedWidth = inputWidth + 2 * padding;

    for (int c = 0; c < outChannels; ++c) {
      //每层channel添加padding
      cv::Mat channelMat = inputData.row(c).reshape(0, inputHeight);
      cv::Mat paddedChannelMat;
      cv::copyMakeBorder(channelMat, paddedChannelMat, padding, padding, padding, padding, cv::BORDER_CONSTANT, 0);
      //滑窗
      for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            cv::Rect window(j * stride_w, i * stride_h, kernelWidth, kernelHeight);
            cv::Mat windowMat = paddedChannelMat(window);
            if (poolType == MAXPOOL) { //max  pooling
                double minVal, maxVal;
                cv::Point maxLoc;
                cv::minMaxLoc(windowMat, &minVal, &maxVal, nullptr, &maxLoc);
                y.at<float>(c, i * outputWidth + j) = static_cast<float>(maxVal);
                max_position.at<cv::Vec2i>(c, i * outputWidth + j) = cv::Vec2i(maxLoc.x + j * stride_w - padding, maxLoc.y + i * stride_h - padding);
            } else if (poolType == AVGPOOL) { // average pooling
                cv::Scalar avgVal = cv::mean(windowMat);
                y.at<float>(c, i * outputWidth + j) = static_cast<float>(avgVal[0]);
            }
        }
      }
    }
}

//backward section

/*
矩阵上采样
如果是maxpooling模式，则把局域梯度放到池化前最大值的位置
5 9        5 0 0 9
     -->   0 0 0 0
3 6        0 0 0 0
           3 0 0 6
如果是average pooling模式，则把局域梯度除以池化窗口的尺寸2*2=4:
5 9        1.25 1.25 2.25 2.25
     -->   1.25 1.25 2.25 2.25
3 6        0.75 0.75 1.5  1.5
           0.75 0.75 1.5  1.5
*/

void PoolingLayer2D::backward(const cv::Mat& d0)
{
    cv::Mat d = d0.reshape(0, outChannels);
    for (int c = 0; c < outChannels; ++c) {
            // 遍历输出梯度的每个元素
      for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
          //max pooling
          if (poolType == MAXPOOL) {
            cv::Vec2i pos = max_position.at<cv::Vec2i>(c, i * outputWidth + j); //get pos recorded
            int idx = pos[1] * inputWidth + pos[0]; // 转换回二维索引
            dx.at<float>(c, idx) = d.at<float>(c, i * outputWidth + j);

          } else if (poolType == AVGPOOL) {
            //average pooling
            for (int m = 0; m < kernelHeight; ++m) {
              for (int n = 0; n < kernelWidth; ++n) {
                int row = i * stride_h + m - padding;
                int col = j * stride_w + n - padding;
                if (row >= 0 && row < inputHeight && col >= 0 && col < inputWidth) {
                    int idx = row * inputWidth + col; // 转换回二维索引
                    dx.at<float>(c, idx) += d.at<float>(c, i * outputWidth + j) / (kernelHeight * kernelWidth);
                }
              }
            }
          }
        }
      }
    }
}

void PoolingLayer2D::zeroGrad(){
    dx = cv::Mat::zeros(inChannels, inputHeight * inputWidth, CV_32F);
}

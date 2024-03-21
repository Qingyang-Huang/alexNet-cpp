#include "poolingLayer.h"

PoolingLayer::PoolingLayer(int inputWidth, int inputHeight, int kernelSize, int stride, int padding, int inChannels, int outChannels, int poolType = MAXPOOL)
    :inputWidth(inputWidth), inputHeight(inputHeight), kernelSize(kernelSize), stride(stride), padding(padding), inChannels(inChannels), outChannels(outChannels), poolType(poolType){
    

    outputWidth = ((inputWidth - kernelSize + 2 * padding) / stride) + 1;
    outputHeight = ((inputHeight - kernelSize + 2 * padding) / stride) + 1;

    y = cv::Mat::zeros(outChannels, outputHeight * outputWidth, CV_32F);
    d = cv::Mat::zeros(outChannels, outputHeight * outputWidth, CV_32F);
    
    max_position = cv::Mat::zeros(outChannels, outputHeight * outputWidth, CV_32SC2); // 存储位置(x, y)

    dx = cv::Mat::zeros(inChannels, inputHeight * inputWidth, CV_32F); 

}

PoolingLayer::~PoolingLayer(){
    delete d;
    delete y;
    delete max_position;
    delete dx;
}

//forwad section 
void PoolingLayer::forward(cv::Mat &inputData)
{
  int paddedHeight = inputHeight + 2 * padding;
  int paddedWidth = inputWidth + 2 * padding;
  int rows = (paddedHeight - kernelSize) / stride + 1;
  int cols = (paddedWidth - kernelSize) / stride + 1;
  //分channel处理
  for (int c = 0; c < outChannels; ++c) {
    //每层channel添加padding
    cv::Mat channelMat = input.row(c).reshape(0, inputHeight);
    cv::Mat paddedChannelMat;
    cv::copyMakeBorder(channelMat, paddedChannelMat, padding, padding, padding, padding, cv::BORDER_CONSTANT, 0);
    //滑窗
    for (int i = 0; i < outputRows; ++i) {
      for (int j = 0; j < outputCols; ++j) {
          cv::Rect window(j * stride, i * stride, kernelSize, kernelSize);
          cv::Mat windowMat = paddedChannelMat(window);
          if (poolType == MAXPOOL) { //max  pooling
              double minVal, maxVal;
              cv::Point maxLoc;
              cv::minMaxLoc(windowMat, &minVal, &maxVal, nullptr, &maxLoc);
              y.at<float>(c, i * outputCols + j) = static_cast<float>(maxVal);
              max_position.at<cv::Vec2i>(c, i * outputCols + j) = cv::Vec2i(maxLoc.x + j * stride - pad, maxLoc.y + i * stride - pad);
          } else if (poolType == AVGPOOL) { // average pooling
              cv::Scalar avgVal = cv::mean(windowMat);
              y.at<float>(c, i * outputCols + j) = static_cast<float>(avgVal[0]);
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

void PoolingLayer::backward(const cv::Mat& d0)
{
  for (int c = 0; c < outChannels; ++c) {
          // 遍历输出梯度的每个元素
    for (int i = 0; i < outputHeight; ++i) {
      for (int j = 0; j < outputWidth; ++j) {
        //max pooling
        if (poolType == MAXPOOL) {
          cv::Vec2i pos = max_position.at<cv::Vec2i>(c, i * outputWidth + j); //get pos recorded
          int idx = pos[1] * inputWidth + pos[0]; // 转换回二维索引
          dx.at<float>(c * inputHeight * inputWidth + idx) = d0.at<float>(c, i * outputWidth + j);

        } else if (poolType == AVGPOOL) {
          //average pooling
          for (int m = 0; m < kernelSize; ++m) {
            for (int n = 0; n < kernelSize; ++n) {
              int row = i * stride + m - padding;
              int col = j * stride + n - padding;
              if (row >= 0 && row < inputHeight && col >= 0 && col < inputWidth) {
                  int idx = row * inputWidth + col; // 转换回二维索引
                  dx.at<float>(c * inputHeight * inputWidth + idx) += d0.at<float>(c, i * outputWidth + j) / (kernelSize * kernelSize);
              }
            }
          }
        }
      }
    }
  }
}

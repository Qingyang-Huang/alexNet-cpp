#include "poolingLayer2D.h"
#include <iostream>

PoolingLayer2D::PoolingLayer2D(int inputWidth, int inputHeight, int kernelHeight, int kernelWidth, int stride_h, int stride_w, int pad_w, int pad_h, int inChannels, int outChannels, int poolType)
  :inputWidth(inputWidth), inputHeight(inputHeight), kernelHeight(kernelHeight), kernelWidth(kernelWidth), stride_h(stride_h), stride_w(stride_w), 
  pad_h(pad_h), pad_w(pad_w), inChannels(inChannels), outChannels(outChannels), poolType(poolType),
  outputWidth(((inputWidth - kernelWidth + 2 * pad_w) / stride_w) + 1),
  outputHeight(((inputHeight - kernelHeight + 2 * pad_h) / stride_h) + 1),
  z({static_cast<size_t>(outChannels), static_cast<size_t>(outputHeight * outputWidth)}),
  dx({static_cast<size_t>(inChannels), static_cast<size_t>(inputHeight * inputWidth)}),
  max_position({static_cast<size_t>(outChannels), static_cast<size_t>(outputHeight * outputWidth * 2)}){}

PoolingLayer2D::~PoolingLayer2D(){}

//forwad section 
void PoolingLayer2D::maxPooling(const Tensor<float> &inputData){
    for (int c = 0; c < outChannels; ++c) {
        for (int i = 0; i < outputHeight; ++i) {
            for (int j = 0; j < outputWidth; ++j) {
              float maxVal = -FLT_MAX;
              int maxLoc[2] = {-1, -1}; // 初始化为无效位置
              for (int m = 0; m < kernelHeight; ++m) {
                  for (int n = 0; n < kernelWidth; ++n) {
                      int row = i * stride_h + m - pad_h;
                      int col = j * stride_w + n - pad_w;
                      if (row >= 0 && row < inputHeight && col >= 0 && col < inputWidth) {
                          int idx = row * inputWidth + col;
                          float tmp = inputData(c, idx);
                          if(tmp > maxVal){
                              maxVal = tmp;
                              maxLoc[0] = row;
                              maxLoc[1] = col;
                          }
                      }else{
                        if(0 > maxVal){
                            maxVal = 0;
                            maxLoc[0] = row;
                            maxLoc[1] = col;
                        }
                      }
                  }
              }
              z(c, i * outputWidth + j) = maxVal;
              max_position(c, (i * outputWidth + j) * 2, 0) = maxLoc[0];
              max_position(c, (i * outputWidth + j) * 2, 1) = maxLoc[1];
          }
        }
    }
}

void PoolingLayer2D::avgPooling(const Tensor<float> &inputData){
    for (int c = 0; c < outChannels; ++c) {
        for (int i = 0; i < outputHeight; ++i) {
            for (int j = 0; j < outputWidth; ++j) {
                float sum = 0;
                for (int m = 0; m < kernelHeight; ++m) {
                    for (int n = 0; n < kernelWidth; ++n) {
                        int row = i * stride_h + m - pad_h;
                        int col = j * stride_w + n - pad_w;
                        if (row >= 0 && row < inputHeight && col >= 0 && col < inputWidth) {
                            int idx = row * inputWidth + col;
                            sum += inputData(c, idx);
                        } else {
                            sum += 0;
                        }
                    }
                }
                z(c, i * outputWidth + j) = sum / (kernelHeight * kernelWidth);
            }
        }
    }
}

void PoolingLayer2D::forward(const Tensor<float> &inputData)
{
    int paddedHeight = inputHeight + 2 * pad_h;
    int paddedWidth = inputWidth + 2 * pad_w;

    if (poolType == MAXPOOL) { //max  pooling
        maxPooling(inputData);
    } else if (poolType == AVGPOOL) {
        avgPooling(inputData); // average pooling
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

void PoolingLayer2D::backward(const Tensor<float>& d0)
{   
    Tensor<float> d = d0.reshape(outChannels, outputHeight * outputWidth, 1);
    for (int c = 0; c < outChannels; ++c) {
            // 遍历输出梯度的每个元素
      for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
          //max pooling
          if (poolType == MAXPOOL) {
            size_t pos_0 = max_position(c, (i * outputWidth + j) * 2, 0);
            size_t pos_1 = max_position(c, (i * outputWidth + j) * 2, 1);
            size_t idx = pos_0 * inputWidth + pos_1; // 转换回二维索引
            dx(c, idx) = d(c, i * outputWidth + j);

          } else if (poolType == AVGPOOL) {
            //average pooling
            for (int m = 0; m < kernelHeight; ++m) {
              for (int n = 0; n < kernelWidth; ++n) {
                int row = i * stride_h + m - pad_h;
                int col = j * stride_w + n - pad_w;
                if (row >= 0 && row < inputHeight && col >= 0 && col < inputWidth) {
                    int idx = row * inputWidth + col; 
                    dx(c, idx) += d(c, i * outputWidth + j) / (kernelHeight * kernelWidth);
                }
              }
            }
          }
        }
      }
    }
}

void PoolingLayer2D::zeroGrad(){
    dx.zeros();
}

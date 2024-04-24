#include "poolingLayer2D.h"
#include <iostream>

PoolingLayer2D::PoolingLayer2D(int inputWidth, int inputHeight, int kernelHeight, int kernelWidth, int stride_h, int stride_w, int padding, int inChannels, int outChannels, int poolType)
  :inputWidth(inputWidth), inputHeight(inputHeight), kernelHeight(kernelHeight), kernelWidth(kernelWidth), stride_h(stride_h), stride_w(stride_w), 
  padding(padding), inChannels(inChannels), outChannels(outChannels), poolType(poolType),
  outputWidth(((inputWidth - kernelWidth + 2 * padding) / stride_w) + 1),
  outputHeight(((inputHeight - kernelHeight + 2 * padding) / stride_h) + 1),
  z({static_cast<size_t>(outChannels), static_cast<size_t>(outputHeight) * outputWidth}),
  dx({static_cast<size_t>(inChannels), static_cast<size_t>(inputHeight) * inputWidth}),
  max_position({static_cast<size_t>(outChannels), static_cast<size_t>(outputHeight) * outputWidth}){}

PoolingLayer2D::~PoolingLayer2D(){}

//forwad section 
void PoolingLayer2D::forward(const Tensor<float> &inputData)
{
    int paddedHeight = inputHeight + 2 * padding;
    int paddedWidth = inputWidth + 2 * padding;

    for (int c = 0; c < outChannels; ++c) {
        for (int i = 0; i < outputHeight; ++i) {
            for (int j = 0; j < outputWidth; ++j) {
                if (poolType == MAXPOOL) { //max  pooling
                    float maxVal = -FLT_MAX;
                    int maxLoc[2];
                    for (int m = 0; m < kernelHeight; ++m) {
                        for (int n = 0; n < kernelWidth; ++n) {
                            int row = i * stride_h + m - padding;
                            int col = j * stride_w + n - padding;
                            float tmp;
                            if (row >= 0 && row < inputHeight && col >= 0 && col < inputWidth) {
                                int idx = row * inputWidth + col;
                                tmp = inputData(c, idx);
                            } else {
                                tmp = 0;
                            }
                            if(tmp > maxVal){
                                maxVal = tmp;
                                maxLoc[0] = row;
                                maxLoc[1] = col;
                            }
                            
                        }
                    }
                    z(c, i * outputWidth + j) = maxVal;
                    max_position(c, i * outputWidth + j, 0) = maxLoc[0];
                    max_position(c, i * outputWidth + j, 1) = maxLoc[1];
                } else if (poolType == AVGPOOL) { // average pooling
                    float sum = 0;
                    for (int m = 0; m < kernelHeight; ++m) {
                        for (int n = 0; n < kernelWidth; ++n) {
                            int row = i * stride_h + m - padding;
                            int col = j * stride_w + n - padding;
                            float tmp;
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
    for (int c = 0; c < outChannels; ++c) {
            // 遍历输出梯度的每个元素
      for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
          //max pooling
          if (poolType == MAXPOOL) {
            int pos_0 = max_position(c, i * outputWidth + j, 0);
            int pos_1 = max_position(c, i * outputWidth + j, 1);
            int idx = pos_1 * inputWidth + pos_0; // 转换回二维索引
            dx(c, idx) = d0(c, i * outputWidth + j);

          } else if (poolType == AVGPOOL) {
            //average pooling
            for (int m = 0; m < kernelHeight; ++m) {
              for (int n = 0; n < kernelWidth; ++n) {
                int row = i * stride_h + m - padding;
                int col = j * stride_w + n - padding;
                if (row >= 0 && row < inputHeight && col >= 0 && col < inputWidth) {
                    int idx = row * inputWidth + col; 
                    dx(c, idx) += d0(c, i * outputWidth + j) / (kernelHeight * kernelWidth);
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

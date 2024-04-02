#include "convolutionalLayer2D.h"
#include <iostream>
#include <random>

ConvolutionalLayer2D::ConvolutionalLayer2D(int inputWidth, int inputHeight, int kernelHeight, int kernelWidth,  int stride_h, int stride_w, int padding, int inChannels, int outChannels, bool useBias)
    : inputWidth(inputWidth), inputHeight(inputHeight), kernelHeight(kernelHeight), kernelWidth(kernelWidth), stride_h(stride_h), stride_w(stride_w), padding(padding),
      inChannels(inChannels), outChannels(outChannels), useBias(useBias){
    // Calculate output dimensions
    outputWidth = ((inputWidth - kernelWidth + 2 * padding) / stride_w) + 1;
    outputHeight = ((inputHeight - kernelHeight + 2 * padding) / stride_h) + 1;
    
    // Initialize matrices
    y = cv::Mat::zeros(outChannels, outputHeight * outputWidth, CV_32F);
    d = cv::Mat::zeros(outChannels, outputHeight * outputWidth, CV_32F);
    dx = cv::Mat::zeros(inChannels, inputHeight * inputWidth, CV_32F);
    bias = cv::Mat::zeros(1, outChannels, CV_32F);

    // srand(time(nullptr)); //general version

    //device version
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // -1 ~ 1

    for (int o = 0; o < outChannels; ++o) {
        for (int i = 0; i < inChannels; ++i) {
            // 创建一个大小为 kernelHeight x kernelWidth 的矩阵，元素类型为 float
            cv::Mat kernelMat(kernelHeight, kernelWidth, CV_32F);

            // 用随机数填充卷积核-1 to 1
            // cv::randu(kernelMat, cv::Scalar::all(-1), cv::Scalar::all(1));  //general method
            for (int y = 0; y < kernelMat.rows; ++y) {
                for (int x = 0; x < kernelMat.cols; ++x) {
                    kernelMat.at<float>(y, x) = dist(gen); // 生成-1到1之间的随机数
                }
            }

            // 将这个卷积核添加到 kernel 向量中
            kernel.push_back(kernelMat);
        }
    }
    
    
}

ConvolutionalLayer2D::~ConvolutionalLayer2D() {}

cv::Mat ConvolutionalLayer2D::conv2D(const cv::Mat& inputData, cv::Mat& kernel) {
    // kernel是一个4D矩阵[outChannels, inChannels, kernelHeight, kernelWidth]
    // inputData是一个3D矩阵[inChannels, inputHeight, inputWidth]

    cv::Mat exInputData;
    cv::Mat reshapedInputData = inputData.reshape(0, inputHeight);
    cv::copyMakeBorder(reshapedInputData, exInputData, padding, padding, padding, padding, cv::BORDER_CONSTANT, 0);

    cv::Mat outputData = cv::Mat::zeros(1, outputHeight * outputWidth, CV_32F);
    
    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernelHeight; ++ky) {
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    int iy = y * stride_h + ky;
                    int ix = x * stride_w + kx;
                    if (iy >= 0 && ix >= 0 && iy < exInputData.rows && ix < exInputData.cols) {
                        sum += exInputData.at<float>(iy, ix) * kernel.at<float>(ky, kx);
                    }
                }
            }
            outputData.at<float>(0, y * outputWidth + x) = sum;
        }
    }
    return outputData;
}

void ConvolutionalLayer2D::forward(const cv::Mat inputData) {
    // 假设mapData、v、y和bias已经正确初始化
    for (int i = 0; i < outChannels; ++i) {
        cv::Mat accumulated = cv::Mat::zeros(1, outputWidth * outputHeight, CV_32F); // 初始化累加器
        for (int j = 0; j < inChannels; ++j) {
            int kernelIndex = i * inChannels + j;
            cv::Mat mapout = conv2D(inputData.row(j), kernel[kernelIndex]);
            accumulated = accumulated + mapout; // 累加所有输入通道的卷积结果
        }
        // 应用激活函数
        for (int c = 0; c < accumulated.cols; ++c) {
            y.at<float>(i, c) = relu(accumulated.at<float>(0, c) + bias.at<float>(0, i)); // 加上偏置
        }    
    }
}

cv::Mat ConvolutionalLayer2D::transConv2D(const cv::Mat& input, const cv::Mat& kernel) {
    // 创建一个扩展后的input矩阵，初始填充为0
    //矩阵两边填充（k-p-1）个padding
    //中间填充(s-1)*(i-1)个padding
    //extend size = (i-1)*(s-1)+2*(k-p-1)
    cv::Mat reshapedInputData = input.reshape(0, outputWidth);
    int expandedHeight = (reshapedInputData.rows - 1) * stride_h + 2 * (kernel.rows - padding) - 1;
    int expandedWidth = (reshapedInputData.cols - 1) * stride_w + 2 * (kernel.cols - padding) - 1;
    
    cv::Mat expandedInput = cv::Mat::zeros(expandedHeight, expandedWidth, input.type());
    // 将input中的元素间隔地复制到expandedInput中
    for (int y = 0; y < reshapedInputData.rows; ++y) {
        for (int x = 0; x < reshapedInputData.cols; ++x) {
            expandedInput.at<float>(y * stride_h + (kernel.rows - padding - 1), x * stride_w + (kernel.cols - padding - 1)) = reshapedInputData.at<float>(y,x);
        }
    }

    cv::Mat output = cv::Mat::zeros(inputHeight, inputWidth, input.type());

    // 对于每个expandedInput元素
    for (int y = 0; y < expandedHeight - kernel.rows + 1; ++y) {
        for (int x = 0; x < expandedWidth - kernel.cols + 1; ++x) {
            // 定义卷积操作的感受野
            cv::Rect roi(x, y, kernel.cols, kernel.rows);
            cv::Mat inputROI = expandedInput(roi);
            // 对于卷积核中的每个元素
            for (int i = 0; i < kernel.rows; ++i) {
                for (int j = 0; j < kernel.cols; ++j) {
                    // 更新输出矩阵的对应位置
                    // kernel.at<float>(kernel.rows - 1 - i, kernel.cols - 1 - j)计算的是反转后的kernel
                    output.at<float>(y, x) += inputROI.at<float>(i, j) * kernel.at<float>(kernel.rows - 1 - i, kernel.cols - 1 - j);
                
                }
            }
        }
    }

    return output.reshape(0,1);
}

void ConvolutionalLayer2D::backward(const cv::Mat& d0) {
    for (int r = 0; r < outputHeight; ++r){
        for (int c = 0; c < outputWidth; ++c){
            for (int i = 0; i < outChannels; ++i){
                d.at<float>(i, r*outputHeight+c) = d0.at<float>(i, r*outputHeight+c) * reluDerivative(y.at<float>(i, r*outputHeight+c));
            }
        }
    }
    for (int i = 0; i < inChannels; ++i){
        for (int j = 0; j < outChannels; ++j){
            int kernelIndex = j * inChannels + i;
            cv::Mat dKernel = transConv2D(d.row(j), kernel[kernelIndex]); //update gradient
            //累加到输入的梯度上
            dx.row(i) =  dx.row(i) + dKernel;
        }
    }
}

void ConvolutionalLayer2D::updateWeight(const cv::Mat& input, float learningRate) {
    for (int i = 0; i < outChannels; ++i){
        for (int j = 0; j < inChannels; ++j){
            int kernelIndex = i * inChannels + j;
            kernel[kernelIndex] = kernel[kernelIndex] - learningRate * d.at<float>(i) * input.at<float>(j);
        }
        if(useBias){
            bias.at<float>(i) = bias.at<float>(i) - learningRate * d.at<float>(i);
        }
        
    }
}

void ConvolutionalLayer2D::zeroGrad(){
    d = cv::Mat::zeros(outputHeight, outputWidth, CV_32F);
    dx = cv::Mat::zeros(inputHeight, inputWidth, CV_32F);
}
    
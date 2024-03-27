#include "convolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(int inputWidth, int inputHeight, int kernelHeight, int kernelWidth, int padding, int stride_h, int stride_w, int inChannels, int outChannels)
    : inputWidth(inputWidth), inputHeight(inputHeight), kernelHeight(kernelHeight), kernelWidth(kernelWidth), padding(padding), stride_h(stride_h), stride_w(stride_w),
      inChannels(inChannels), outChannels(outChannels){
    // Calculate output dimensions
    outputWidth = ((inputWidth - kernelWidth + 2 * padding) / stride_w) + 1;
    outputHeight = ((inputHeight - kernelHeight + 2 * padding) / stride_h) + 1;
    
    // Initialize matrices
    y = cv::Mat::zeros(outChannels, outputHeight * outputWidth, CV_32F);
    d = cv::Mat::zeros(outChannels, outputHeight * outputWidth, CV_32F);
    bias = cv::Mat::zeros(1, outChannels, CV_32F);
    dx = cv::Mat::zeros(inChannels, inputHeight * inputWidth, CV_32F);
    // Initialize kernel with random values
    // kernel = cv::Mat(outChannels, inChannels * kernelSize * kernelSize, CV_32F);
    // cv::randu(kernel, cv::Scalar::all(-1), cv::Scalar::all(1)); // Random values in range [-1, 1]

    srand(time(nullptr));

        // 为每个输入通道到每个输出通道的组合创建一个卷积核
    for (int o = 0; o < outChannels; ++o) {
        for (int i = 0; i < inChannels; ++i) {
            // 创建一个大小为 kernelHeight x kernelWidth 的矩阵，元素类型为 float
            cv::Mat kernelMat(kernelHeight, kernelWidth, CV_32F);

            // 用随机数填充卷积核-1 to 1
            cv::randu(kernelMat, cv::Scalar::all(-1), cv::Scalar::all(1));

            // 将这个卷积核添加到 kernel 向量中
            kernel.push_back(kernelMat);
        }
    }
    
    
}

ConvolutionalLayer::~ConvolutionalLayer() {}

cv::Mat ConvolutionalLayer::conv2D(const cv::Mat& inputData, cv::Mat& kernel) {
    // kernel是一个4D矩阵[outChannels, inChannels, kernelHeight, kernelWidth]
    // inputData是一个3D矩阵[inChannels, inputHeight, inputWidth]

    cv::Mat exInputData;
    cv::copyMakeBorder(inputData, exInputData, padding, padding, padding, padding, cv::BORDER_CONSTANT, 0);

    int outputHeight = ((exInputData.rows - kernelHeight + 2 * padding) / stride_h) + 1;
    int outputWidth = ((exInputData.cols - kernelWidth + 2 * padding) / stride_w) + 1;

    cv::Mat OutputData = cv::Mat::zeros(outputHeight, outputWidth, CV_32F);

    for (int y = 0; y < outputHeight; ++y) {
        for (int x = 0; x < outputWidth; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernelHeight; ++ky) {
                for (int kx = 0; kx < kernelWidth; ++kx) {
                    int iy = y * stride_h + ky - padding;
                    int ix = x * stride_w + kx - padding;
                    if (iy >= 0 && ix >= 0 && iy < inputData.rows && ix < inputData.cols) {
                        sum += inputData.at<float>(iy, ix) * kernel.at<float>(ky, kx);
                    }
                }
            }
            OutputData.at<float>(y, x) = sum;
        }
    }

    return OutputData;
}

void ConvolutionalLayer::forward(const cv::Mat inputData) {
    // 假设mapData、v、y和bias已经正确初始化
    for (int i = 0; i < outChannels; ++i) {
        cv::Mat accumulated = cv::Mat::zeros(outputHeight, outputWidth, CV_32F); // 初始化累加器

        for (int j = 0; j < inChannels; ++j) {
            int kernelIndex = i * inChannels + j;
            cv::Mat mapout = conv2D(inputData, kernel[kernelIndex]);
            accumulated = accumulated + mapout; // 累加所有输入通道的卷积结果
        }

        // 应用激活函数
        for (int r = 0; r < accumulated.rows; ++r) {
            for (int c = 0; c < accumulated.cols; ++c) {
                y.at<cv::Mat>(i).at<float>(r, c) = relu(accumulated.at<float>(r, c) + bias.at<float>(0, i)); // 加上偏置
            }
        }
        
    }
}

cv::Mat ConvolutionalLayer::transConv2D(const cv::Mat& input, const cv::Mat& kernel) {
    // 创建一个扩展后的input矩阵，初始填充为0
    //矩阵两边填充（k-p-1）个padding
    //中间填充(s-1)*(i-1)个padding
    //extend size = (i-1)*(s-1)+2*(k-p-1)

    int expandedHeight = (input.rows - 1) * (stride_h - 1) + 2 * (kernel.rows - padding - 1);
    int expandedWidth = (input.cols - 1) * (stride_w - 1) + 2 * (kernel.cols - padding - 1);
    
    cv::Mat expandedInput = cv::Mat::zeros(expandedHeight, expandedWidth, input.type());
    // 将input中的元素间隔地复制到expandedInput中
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            expandedInput.at<float>(y * stride_h + (kernel.rows - padding - 1), x * stride_w + (kernel.cols - padding - 1)) = input.at<float>(y, x);
        }
    }

    // 初始化输出矩阵
    cv::Mat output = cv::Mat::zeros(outputHeight, outputWidth, input.type());

    // 对于每个expandedInput元素
    for (int y = 0; y < inputHeight - kernel.rows + 1; ++y) {
        for (int x = 0; x < inputWidth - kernel.cols + 1; ++x) {
            // 定义卷积操作的感受野
            cv::Rect roi(x, y, kernel.cols, kernel.rows);
            cv::Mat inputROI = expandedInput(roi);

            // 对于卷积核中的每个元素
            for (int i = 0; i < kernel.rows; ++i) {
                for (int j = 0; j < kernel.cols; ++j) {
                    // 更新输出矩阵的对应位置
                    int outY = y + i;
                    int outX = x + j;
                    if (outY < outputHeight && outX < outputWidth) {
                        output.at<float>(outY, outX) += inputROI.at<float>(i, j) * kernel.at<float>(kernel.rows - 1 - i, kernel.cols - 1 - j);
                    }
                }
            }
        }
    }

    return output;
}

void ConvolutionalLayer::backward(const cv::Mat& d0) {
    for (int r = 0; r < outputHeight; ++r){
        for (int c = 0; c < outputWidth; ++c){
            for (int i = 0; i < outChannels; ++i){
                d.at<cv::Mat>(i).at<float>(r, c) = d0.at<cv::Mat>(i).at<float>(r, c) * reluDerivative(y.at<cv::Mat>(i).at<float>(r, c));
            }
        }
    }
    for (int i = 0; i < outChannels; ++i){
        for (int j = 0; j < inChannels; ++j){
            int kernelIndex = i * inChannels + j;
            cv::Mat flipKernel;
            cv::flip(kernel[kernelIndex], flipKernel, -1);   //卷积核先顺时针旋转180度
            cv::Mat dKernel = transConv2D(d.at<cv::Mat>(i), flipKernel); //update gradient
            dx.at<cv::Mat>(j) = dx.at<cv::Mat>(j) + dKernel; // 累加到输入的梯度上
        }
    }
}

void ConvolutionalLayer::updateWeight(const cv::Mat& input, float learningRate) {
    for (int i = 0; i < outChannels; ++i){
        for (int j = 0; j < inChannels; ++j){
            int kernelIndex = i * inChannels + j;
            kernel[kernelIndex] = kernel[kernelIndex] - learningRate * d.at<float>(i) * input.at<float>(j);
        }
        bias.at<float>(i) = bias.at<float>(i) - learningRate * d.at<float>(i);
    }
}

void ConvolutionalLayer::zeroGrad(){
    d = cv::Mat::zeros(outputHeight, outputWidth, CV_32F);
    dx = cv::Mat::zeros(inputHeight, inputWidth, CV_32F);
}
    
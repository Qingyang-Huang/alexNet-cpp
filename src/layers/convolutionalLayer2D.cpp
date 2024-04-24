#include "convolutionalLayer2D.h"
#include <iostream>
#include <random>

ConvolutionalLayer2D::ConvolutionalLayer2D(int inputHeight, int inputWidth, int kernelHeight, int kernelWidth,  int stride_h, int stride_w, int pad_h, int pad_w, int inChannels, int outChannels, bool useBias)
    : inputHeight(inputHeight), inputWidth(inputWidth), kernelHeight(kernelHeight), kernelWidth(kernelWidth), stride_h(stride_h), stride_w(stride_w), pad_h(pad_h), pad_w(pad_w),
        inChannels(inChannels), outChannels(outChannels), useBias(useBias),
        outputWidth(((inputWidth - kernelWidth + 2 * pad_w) / stride_w) + 1),
        outputHeight(((inputHeight - kernelHeight + 2 * pad_h) / stride_h) + 1),
        kernels({static_cast<size_t>(outChannels), static_cast<size_t>(inChannels * kernelHeight * kernelWidth)}),
        bias(static_cast<size_t>(outChannels)),
        z({static_cast<size_t>(outChannels), static_cast<size_t>(outputHeight) * outputWidth}),
        a({static_cast<size_t>(outChannels), static_cast<size_t>(outputHeight) * outputWidth}),
        d({static_cast<size_t>(outChannels), static_cast<size_t>(outputHeight) * outputWidth}),
        dx({static_cast<size_t>(inChannels), static_cast<size_t>(inputHeight) * inputWidth})
      {
    //device version
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < outChannels; ++i) {
        for (int j = 0; j < inChannels; ++j) {
            for (int y = 0; y < kernelHeight; ++y) {
                for (int x = 0; x < kernelWidth; ++x) {
                    int kernelIndex = y * inChannels + x;
                    kernels(i, j, kernelIndex) = dist(gen);
                }
            }
        }
        if (useBias) {
            bias(i) = dist(gen); 
        }
    }
}

ConvolutionalLayer2D::~ConvolutionalLayer2D() {}

Tensor<float> ConvolutionalLayer2D::img2col(const Tensor<float>& imgData) {
    //img size [inChannels, inputHeight, inputWidth]
    //col size [inChannels * kernelHeight * kernelWidth, outputHeight * outputWidth]
    int colHeight = inChannels * kernelHeight * kernelWidth;
    int colWidth = outputHeight * outputWidth;
    Tensor<float> colData = Tensor<float>(colHeight, colWidth);

    for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            for (int c = 0; c < inChannels; ++c) {
                for (int y = 0; y < kernelHeight; ++y) {
                    for (int x = 0; x < kernelWidth; ++x) {
                        int colRow = i * outputWidth + j;
                        int colCol = c * kernelHeight * kernelWidth + y * kernelWidth + x;
                        
                        // Calculate the input image index considering stride and kernel size
                        int inputY = i * stride_h - pad_h + y;
                        int inputX = j * stride_w - pad_w + x;
                        
                        if (inputY >= 0 && inputY < inputHeight && inputX >= 0 && inputX < inputWidth) {
                            // Valid index within image bounds
                            int imgRow = c * inputHeight * inputWidth + inputY * inputWidth + inputX;
                            colData(colCol, colRow) = imgData(imgRow);
                        } else {
                            // Out-of-bounds indices get zero padding
                            colData(colCol, colRow) = 0;
                        }
                    }
                }
            }
        }
    }
    return colData;
}


void ConvolutionalLayer2D::conv2D(const Tensor<float>& inputData) {
    // kernels [outChannels, inChannels * kernelHeight * kernelWidth]
    // inputData [inChannels * kernelHeight * kernelWidth, outputHeight * outputWidth]
    // kernels.resize(outChannels, inChannels * kernelHeight * kernelWidth);
    
    // Tensor paddedInputData = Tensor<float>(inChannels, paddedHeight * paddedWidth);
    // for (int c = 0; c < inChannels; ++c) {
    //     for (int h = 0; h < inputHeight; ++h) {
    //         for (int w = 0; w < inputWidth; ++w) {
    //             paddedInputData(c, (h + pad_h) * paddedHeight + w + pad_w) = inputData(c, h * inputWidth + w);
    //         }
    //     }
    // }
    
    // paddedInputData.reshape(inChannels, paddedHeight, paddedWidth).print();
    Tensor colData = img2col(inputData);
    // colData.reshape(1, inChannels * kernelHeight * kernelWidth, outputHeight * outputWidth).print();
    z = kernels.dot(colData);
    //tmp size [outChannels, outputHeight * outputWidth]
    // z.reshape(1, outChannels, outputHeight * outputWidth).print();
    if(useBias){
        for (int o = 0; o < outChannels; ++o) {
            for(int idx = 0; idx < outputHeight * outputWidth; ++idx){
                z(o, idx) += bias(o);
            }
        }
    }
}

void ConvolutionalLayer2D::forward(const Tensor<float>& inputData) {
    conv2D(inputData);
    // activation
    for (int i = 0; i < outChannels; ++i){
        for (int j = 0; j < outputHeight * outputWidth; ++j){
            a(i, j) = relu(z(i, j));
        }
    }
}

Tensor<float> ConvolutionalLayer2D::col2img(const Tensor<float>& colData) {
    Tensor<float> imgData = Tensor<float>(inChannels, inputHeight, inputWidth);
    for (int i = 0; i < outputHeight; ++i) {
        for (int j = 0; j < outputWidth; ++j) {
            for (int c = 0; c < inChannels; ++c) {
                for (int y = 0; y < kernelHeight; ++y) {
                    for (int x = 0; x < kernelWidth; ++x) {
                        int colRow = i * outputWidth + j;
                        int colCol = c * kernelHeight * kernelWidth + y * kernelWidth + x;
                        int realY = (i * stride_h + y) - pad_h;
                        int realX = (j * stride_w + x) - pad_w;

                        if (realY >= 0 && realY < inputHeight && realX >= 0 && realX < inputWidth) {
                            int imgRow = c * inputHeight * inputWidth + realY * inputWidth + realX;
                            imgData(imgRow) += colData(colRow, colCol);
                        }
                    }
                }
            }
        }
    }
    return imgData;
}

void ConvolutionalLayer2D::transConv2D(const Tensor<float>& input) {
    Tensor tmp =  input.t().dot(kernels);
    // d.reshape(1, outputHeight * outputWidth, inChannels * kernelHeight * kernelWidth).print();  
    dx = col2img(tmp);
    // dx.reshape(inChannels, inputHeight, inputWidth).print();
}

void ConvolutionalLayer2D::backward(const Tensor<float>& d0) {
    for (int o = 0; o < outChannels; ++o){
        for (int r = 0; r < outputHeight; ++r){
            for (int c = 0; c < outputWidth; ++c){
                int idx = r * outputWidth + c;
                d(o, idx) = d0(o, idx) * reluDerivative(z(o, idx));
                // printf("d0: %f, z: %f, d: %f\n", d0(o, idx), z(o, idx), d(o, idx));
            }
        }
    }
    transConv2D(d);
}

void ConvolutionalLayer2D::updateWeight(const Tensor<float>& al, float learningRate) {
    Tensor weight = d.dot(img2col(al).t());
    img2col(al).reshape(inChannels, kernelHeight * kernelWidth, outputHeight * outputWidth).print();
    weight.reshape(outChannels, inChannels, kernelHeight * kernelWidth).print();
    kernels -= weight * learningRate;
    if(useBias){
        for(int o = 0; o < outChannels; ++o){
            bias(o) -= learningRate * (d.row(o).sum());

        }
    }
}

void ConvolutionalLayer2D::zeroGrad(){
    d.zeros();
    dx.zeros();
}

void ConvolutionalLayer2D::setKernels(const Tensor<float>& kernel){
    if(kernel.size() != kernels.size()){
        std::cerr << "kernel size not match" << std::endl;
        return;
    }
    kernels = kernel;
}

void ConvolutionalLayer2D::setBias(const Tensor<float>& bias){
    if(bias.size() != this->bias.size()){
        std::cerr << "bias size not match" << std::endl;
        return;
    }
    this->bias = bias;
}
    
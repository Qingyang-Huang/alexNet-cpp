#include"alexNet.h"

AlexNet::AlexNet(int inputSize_h, int inputSize_w, int outputSize)
:inputSize_h(inputSize_h), inputSize_w(inputSize_w), outputSize(outputSize){
    //1层
    C1 = new ConvolutionalLayer(inputSize_h, inputSize_w, 11, 11, 2, 4, 4, 3, 96);   //卷积层
    int in_h = C1->getOutputHeight();  
    int in_w = C1->getOutputHeight();
    S2 = new PoolingLayer(in_h, in_w, 3, 3, 2, 2, 0, C1->getOutChannels(), C1->getOutChannels(), MAXPOOL);   //池化层

    //2层
    in_h = S2->getOutputHeight();
    in_w = S2->getOutputHeight();
    C3 = new ConvolutionalLayer(in_h, in_w, 5, 5, 1, 1, 2, S2->getOutChannels(), 192);   //卷积层
    in_h = C3->getOutputHeight();
    in_w = C3->getOutputHeight();
    S4 = new PoolingLayer(in_h, in_w, 3, 3, 2, 2, 0, C3->getOutChannels(), C3->getOutChannels(), MAXPOOL);    //池化层

    //3
    in_h = S4->getOutputHeight();
    in_w = S4->getOutputHeight();
    C5 = new ConvolutionalLayer(in_h, in_w, 3, 3, 1, 1, 1, S4->getOutChannels(), 384);   //卷积层
    //4
    in_h = C5->getOutputHeight();
    in_w = C5->getOutputHeight();
    C6 = new ConvolutionalLayer(in_h, in_w, 3, 3, 1, 1, 1, C5->getOutChannels(), 256);   //卷积层
    
    //5
    in_h = C6->getOutputHeight();
    in_w = C6->getOutputHeight();
    C7 = new ConvolutionalLayer(in_h, in_w, 3, 3, 1, 1, 1, C6->getOutChannels(), 256);   //卷积层
    in_h = C7->getOutputHeight();
    in_w = C7->getOutputHeight();
    S8 = new PoolingLayer(in_h, in_w, 3, 3, 2, 2, 0, C7->getOutChannels(), C7->getOutChannels(), MAXPOOL);    //池化层

    //O5层
    in_h = S8->getOutputHeight();
    in_w = S8->getOutputHeight();
    O9 = new OutputLayer(in_h*in_w*S8->getOutChannels(), outputSize);    //输出层

    // e = Mat::zeros(1, O5.getOutputNum(), CV_32FC1);   //输出层的输出值与标签值之差
}

AlexNet::~AlexNet(){
    delete C1;
    delete S2;
    delete C3;
    delete S4;
    delete C5;
    delete C6;
    delete C7;
    delete S8;
    delete O9;
}

void AlexNet::forward(cv::Mat input){
    C1->forward(input);
    S2->forward(C1->getY());
    C3->forward(S2->getY());
    S4->forward(C3->getY());
    C5->forward(S4->getY());
    C6->forward(C5->getY());
    C7->forward(C6->getY());
    S8->forward(C7->getY());
    O9->forward(S8->getY());
}

void AlexNet::backward(cv::Mat input, cv::Mat label){
    // e = O9->getY() - label;
    O9->backward(label);
    S8->backward(O9->getDx());
    C7->backward(S8->getDx());
    C6->backward(C7->getDx());
    C5->backward(C6->getDx());
    S4->backward(C5->getDx());
    C3->backward(S4->getDx());
    S2->backward(C3->getDx());
    C1->backward(S2->getDx());
}

void AlexNet::updateWeight(cv::Mat input, float learningRate){
    O9->updateWeight(S8->getY(), learningRate);
    C7->updateWeight(C6->getY(), learningRate);
    C6->updateWeight(C5->getY(), learningRate);
    C5->updateWeight(S4->getY(), learningRate);
    C3->updateWeight(S2->getY(), learningRate);
    C1->updateWeight(input, learningRate);
}

void AlexNet::zeroGrad(){
    O9->zeroGrad();
    S8->zeroGrad();
    C7->zeroGrad();
    C6->zeroGrad();
    C5->zeroGrad();
    S4->zeroGrad();
    C3->zeroGrad();
    S2->zeroGrad();
    C1->zeroGrad();
}

void AlexNet::train(cv::Mat input, cv::Mat label, float learningRate){
    forward(input);
    backward(input, label);
    zeroGrad();
    updateWeight(input, learningRate);
}


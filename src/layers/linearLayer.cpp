#include "linearLayer.h"
#include <random>

LinearLayer::LinearLayer(int inputNum, int outputNum): inputNum(inputNum), outputNum(outputNum) {
    isFullConnect = true;

    bias = cv::Mat::zeros(1, outputNum, CV_32FC1);    //偏置,分配内存的同时初始化为0
    d = cv::Mat::zeros(1, outputNum, CV_32FC1);
    y = cv::Mat::zeros(1, outputNum, CV_32FC1);

    dx = cv::Mat::zeros(1, inputNum, CV_32FC1);   //gradient to upper layer

    // 权重的初始化
    wData = cv::Mat::zeros(outputNum, inputNum, CV_32FC1);   // 输出行，输入列
    
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for(int i = 0; i < outputNum; i++)
    {
        for(int j = 0; j < inputNum; j++)
        {
            //使用随机数初始化权重
            // float randnum = (((float)rand()/(float)RAND_MAX)-0.5)*2; // 产生一个-1到1的随机数,rand()的取值范围为0~RAND_MAX
            float randnum = dist(gen);
            wData.at<float>(i,j) = randnum*sqrt(6.0/(inputNum+outputNum));
        }
    }
}

LinearLayer::~LinearLayer() {}



void LinearLayer::forward(const cv::Mat& inputData)
{  
    cv::Mat flattenData = inputData.reshape(1, 1); //flatten
    //wData [outputNum, inputNum]  flattenData [1, inputNum]
    //∑in[i]*w[i]
    cv::Mat weightedSum = flattenData * wData.t(); 
    
    //weightSum[1, outputNum]  bias[1, outputNum]
    y = weightedSum + bias;
}

void LinearLayer::backward(const cv::Mat d0)
{
    d = d0;
    dx = d * wData;          
}

void LinearLayer::updateWeight(const cv::Mat& input, float learningRate)
{
    cv::Mat dW = d.t() * input.reshape(0, 1);
    wData = wData - learningRate * dW;
    bias = bias - learningRate * d;
}

void LinearLayer::zeroGrad()
{
    d = cv::Mat::zeros(1, outputNum, CV_32FC1);
    dx = cv::Mat::zeros(1, inputNum, CV_32FC1);
}
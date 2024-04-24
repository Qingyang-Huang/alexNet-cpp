#include "linearLayer.h"
#include <random>

LinearLayer::LinearLayer(int inputNum, int outputNum): inputNum(inputNum), outputNum(outputNum),
    wData(outputNum, inputNum), bias(1, outputNum), d(1, outputNum), z(1, outputNum), dx(1, inputNum){   
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
            wData(i,j) = randnum*sqrt(6.0/(inputNum+outputNum));
        }
        bias(i) = dist(gen);
    }
}

LinearLayer::~LinearLayer() {}



void LinearLayer::forward(const Tensor<float>& inputData)
{  
    Tensor flattenData = inputData.reshape(1, inputNum, 1); //flatten
    //wData [outputNum, inputNum]  flattenData [1, inputNum]
    //∑in[i]*w[i]
    Tensor weightedSum = flattenData * wData.t(); 
    
    //weightSum[1, outputNum]  bias[1, outputNum]
    z = weightedSum + bias;
}

void LinearLayer::backward(const Tensor<float> d0)
{
    d = d0;
    dx = d * wData;          
}

void LinearLayer::updateWeight(const Tensor<float>& input, float learningRate)
{
    Tensor dW = d.t() * input.reshape(1, inputNum, 1); 
    wData = wData - dW * learningRate;
    bias = bias - d * learningRate;
}

void LinearLayer::zeroGrad()
{
    d.zeros();
    dx.zeros();
}
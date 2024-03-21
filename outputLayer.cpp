#include "outputLayer.h"

OutputLayer::OutputLayer(int inputNum, int outputNum): inputNum(inputNum), outputNum(outputNum) {
    isFullConnect = true;

    bias = cv::Mat::zeros(1, outputNum, CV_32FC1);    //偏置,分配内存的同时初始化为0
    d = cv::Mat::zeros(1, outputNum, CV_32FC1);
    v = cv::Mat::zeros(1, outputNum, CV_32FC1);
    y = cv::Mat::zeros(1, outputNum, CV_32FC1);

    dx = cv::Mat::zeros(1, inputNum, CV_32FC1);   //gradient to upper layer

    // 权重的初始化
    wData = cv::Mat::zeros(outputNum, inputNum, CV_32FC1);   // 输出行，输入列,权重为10*192矩阵
    //todo srand
    srand((unsigned)time(NULL));
    for(int i = 0; i < outputNum; i++)
    {
        float *p = wData.ptr<float>(i);
        for(int j = 0; j < inputNum; j++)
        {
            //使用随机数初始化权重
            float randnum = (((float)rand()/(float)RAND_MAX)-0.5)*2; // 产生一个-1到1的随机数,rand()的取值范围为0~RAND_MAX
            p[j] = randnum*sqrt(6.0/(inputNum+outputNum));
        }
    }
}

OutputLayer::~OutputLayer() {
    delete wData;
    delete bias;
}



void OutputLayer::forward(const cv::Mat& inputData)
{  
    cv::Mat flattenData = inputData.reshape(1, 1); //flatten
    //wData [outputNum, inputNum]  flattenData [1, inputNum]
    //∑in[i]*w[i]
    cv::Mat weightedSum = flattenData * wData.t(); 
    
    //weightSum[1, outputNum]  bias[1, outputNum]
    v = weightedSum + bias;
    
    
    //Affine层的输出经过Softmax函数，转换成0~1的输出结果
    //softmax
    y = softmax(v);
}

void OutputLayer::backward(const cv::Mat outputData)
{
    //softMax bp
    cv::subtract(y, outputData, d); // d = y - t

    // 计算权重的梯度和更新dx
    dx = d * wData;
          
}
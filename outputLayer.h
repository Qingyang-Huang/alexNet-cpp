#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H
#include <vector>
#include <opencv2/core.hpp>

class OutputLayer {
public:
    // 构造函数
    OutputLayer(int inputNum, int outputNum);
    // 析构函数
    ~OutputLayer();

    int getInputNum() const { return inputNum; }

    int getOutputNum() const { return outputNum; }

    const cv::Mat& getWData() const { return wData; }

    const cv::Mat& getBasicData() const { return basicData; }

    const cv::Mat& getV() const { return v; }

    const cv::Mat& getY() const { return y; }

    const cv::Mat& getD() const { return d; }

    const cv::Mat& getDx() const { return dx; }

    const cv::Mat& getGrad() const { return dx; }
  

private:
    int inputNum;   //输入数据的数目
    int outputNum;  //输出数据的数目
  
    cv::Mat wData;      // 权重数据，为一个inputNum*outputNum大小
    cv::Mat bias;        //偏置，大小为outputNum大小

    cv::Mat v;     // 进入激活函数的输入值
    cv::Mat y;     // 激活函数后神经元的输出
    cv::Mat d;     // 网络的局部梯度

    cv::Mat dx;  //grad backword to upper layer

    bool isFullConnect;

public:
    void forward(const cv::Mat& inputData);
    void backward(const cv::Mat outputData);
    void updateWeight(const cv::Mat& input, float learningRate);
    void zeroGrad();
};

#endif // OUTPUT_LAYER_H

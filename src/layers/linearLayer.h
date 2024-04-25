#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H
#include <vector>
#include "activation.h"
#include "common/tensor.h"

class LinearLayer {
public:
    // 构造函数
    LinearLayer(int inputNum, int outputNum);
    // 析构函数
    ~LinearLayer();

    int getInputNum() const { return inputNum; }

    int getOutputNum() const { return outputNum; }

    const Tensor<float>& getWData() const { return wData; }

    const Tensor<float>& getBasicData() const { return bias; }

    const Tensor<float>& getZ() const { return z; }

    const Tensor<float>& getD() const { return d; }

    const Tensor<float>& getDx() const { return dx; }
  

private:
    int inputNum;   //输入数据的数目
    int outputNum;  //输出数据的数目
  
    Tensor<float> wData;      // 权重数据，为一个inputNum*outputNum大小
    Tensor<float> bias;        //偏置，大小为outputNum大小

    Tensor<float> z;     // 激活函数后神经元的输出
    Tensor<float> dx;  //grad backword to upper layer
    Tensor<float> d;   //grad backword to this layer

public:
    void forward(const Tensor<float>& inputData);
    void backward(const Tensor<float> outputData);
    void updateWeight(const Tensor<float>& input, float learningRate);
    void zeroGrad();
    void setWeight(const Tensor<float>& w);
    void setBias(const Tensor<float>& b);
};

#endif // OUTPUT_LAYER_H

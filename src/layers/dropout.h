#ifndef DROPOUT_H
#define DROPOUT_H
#include <random>
#include "common/tensor.h"

inline Tensor<float> dropout(Tensor<float> input, float p = 0.5) {
    if(p < 0.0 || p > 1.0) {
        throw std::invalid_argument("dropout probability has to be between 0 and 1");
    }
    Tensor<float> output = input.clone();

    std::random_device rd; // 随机数生成器
    std::mt19937 gen(rd()); // 以随机设备作为种子
    std::bernoulli_distribution d(1 - p); // 以1-p的概率生成true
    for (int i = 0; i < output.shape()[0]; ++i) {
        for (int j = 0; j < output.shape()[1]; ++j) {
            for (int k = 0; k < output.shape()[2]; ++k) {
                if (!d(gen)) {
                    output(i, j, k) = 0.0f;
                }
            }
        }
    }

    return output;
}

#endif // DROPOUT_H
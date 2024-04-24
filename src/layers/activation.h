#ifndef ACTIVATIONLAYER_H
#define ACTIVATIONLAYER_H

#include <cmath>
#include "common/tensor.h"

#define RELU 0
#define SIGMOID 1
#define TANH 2

inline float relu(float x) { return fmax(0.0f, x); }
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float softmax(float x) { return exp(x); }
inline Tensor<float> softmax(const Tensor<float>& v) {
    float maxVal = v.max();

    Tensor<float> exp_v(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        exp_v(i) = std::exp(v(i) - maxVal); // Subtract maxVal for numerical stability
    }

    float sum = exp_v.sum();
    return exp_v / sum;
}


inline Tensor<float> softmaxDerivative(const Tensor<float>& v) {
    Tensor<float> output(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        output(i) = v(i) * (1 - v(i));
    }
    return output;
}

inline float reluDerivative(float x) { return x > 0 ? 1 : 0; }
inline float sigmoidDerivative(float x) { return sigmoid(x) * (1 - sigmoid(x)); }
inline float tanhDerivative(float x) { return 1 - tanh(x) * tanh(x); }

#endif // ACTIVATIONLAYER_H
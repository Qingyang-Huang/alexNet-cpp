#ifndef ACTIVATIONLAYER_H
#define ACTIVATIONLAYER_H

#include <math>

#define RELU 0
#define SIGMOID 1
#define TANH 2

inline float relu(float x) { return max(0.0f, x); }
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float tanh(float x) { return tanh(x); }

inline float reluDerivative(float x) { return x > 0 ? 1 : 0; }
inline float sigmoidDerivative(float x) { return sigmoid(x) * (1 - sigmoid(x)); }
inline float tanhDerivative(float x) { return 1 - tanh(x) * tanh(x); }


inline cv::Mat activation(const cv::Mat inputData, int activationType){
    cv::Mat outputData = cv::Mat::zeros(inputData.rows, inputData.cols, CV_32F);
    for (int i = 0; i < inputData.rows; ++i) {
        for (int j = 0; j < inputData.cols; ++j) {
            switch (activationType) {
                case RELU:
                    outputData.at<float>(i, j) = relu(inputData.at<float>(i, j));
                    break;
                case SIGMOID:
                    outputData.at<float>(i, j) = sigmoid(inputData.at<float>(i, j));
                    break;
                case TANH:
                    outputData.at<float>(i, j) = tanh(inputData.at<float>(i, j));
                    break;
                default:
                    break;
            }
        }
    }
    return outputData;
}

inline cv::Mat de_activation(const cv::Mat inputData, int activationType){
    cv::Mat outputData = cv::Mat::zeros(inputData.rows, inputData.cols, CV_32F);
    for (int i = 0; i < inputData.rows; ++i) {
        for (int j = 0; j < inputData.cols; ++j) {
            switch (activationType) {
                case RELU:
                    outputData.at<float>(i, j) = reluDerivative(inputData.at<float>(i, j));
                    break;
                case SIGMOID:
                    outputData.at<float>(i, j) = sigmoidDerivative(inputData.at<float>(i, j));
                    break;
                case TANH:
                    outputData.at<float>(i, j) = tanhDerivative(inputData.at<float>(i, j));
                    break;
                default:
                    break;
            }
        }
    }
    return outputData;
}



#endif // ACTIVATIONLAYER_H
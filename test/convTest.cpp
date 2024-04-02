#include <gtest/gtest.h>
#include "layers/convolutionalLayer2D.h"
#include <opencv2/core/core.hpp>
#include "testUtils.h"

TEST(ConvolutionalLayer2D, Conv2D) {
    cv::Mat input = (cv::Mat_<float>(1, 3*3) << 1, 2, 3,
                                             4, 5, 6,
                                             7, 8, 9);
    cv::Mat kernel = (cv::Mat_<float>(2, 2) << 1, 0,
                                                0, -1);
    
    ConvolutionalLayer2D convLayer(3, 3, 2, 2, 1, 1, 1, 1, 1); 

    cv::Mat output = convLayer.conv2D(input, kernel);

    printMat(output);
    cv::Mat expected = (cv::Mat_<float>(1, 4 * 4) << -1, -2, -3, 0,
                                                 -4, -4, -4, 3,
                                                 -7, -4, -4, 6,
                                                  0, 7, 8, 9);
    EXPECT_TRUE(matEqual(output, expected));
}

TEST(Convolutional2DLayerTest, Forward) {
    cv::Mat input = (cv::Mat_<float>(1,49) << 1, 2, 3, 1, 2, 3, 1,
                                             4, 5, 6, 4, 5, 6, 4,
                                             1, 2, 3, 1, 2, 3, 1,
                                             4, 5, 6, 4, 5, 6, 4,
                                             1, 2, 3, 1, 2, 3, 1,
                                             4, 5, 6, 4, 5, 6, 4,
                                             1, 2, 3, 1, 2, 3, 1);
    ConvolutionalLayer2D *convLayer;
    convLayer = new ConvolutionalLayer2D(7, 7, 3, 3, 1, 1, 1, 1, 1); 
    // 调用forward方法
    convLayer->forward(input);

    // printMat(convLayer->getY());

    // cv::Mat expectedOutput = (cv::Mat_<float>(1,1) << some_value); 
    // EXPECT_EQ(convLayer->getY().at<cv::Mat>(0).at<float>(0, 0), expectedOutput.at<float>(0, 0));
}

TEST(Convolutional2DLayerTest, Forward2) {
    cv::Mat input = cv::Mat(3, 9, CV_32F);

    cv::randu(input, cv::Scalar::all(1), cv::Scalar::all(10));

    ConvolutionalLayer2D *convLayer;
    convLayer = new ConvolutionalLayer2D(3, 3, 2, 2, 1, 1, 1, 3, 6); 
    // 调用forward方法
    convLayer->forward(input);

    // printMat(convLayer->getY());

    // cv::Mat expectedOutput = (cv::Mat_<float>(1,1) << some_value); 
    // EXPECT_EQ(convLayer->getY().at<cv::Mat>(0).at<float>(0, 0), expectedOutput.at<float>(0, 0));
}

TEST(ConvolutionalLayer2D, TransConv2D) {
    cv::Mat input = (cv::Mat_<float>(1, 3*3) << 1, 2, 3,
                                             4, 5, 6,
                                             7, 8, 9);
    cv::Mat kernel = (cv::Mat_<float>(2, 2) << 1, 0,
                                                0, -1);
    
    ConvolutionalLayer2D convLayer(4, 4, 2, 2, 2, 2, 1, 1, 1); 

    cv::Mat output = convLayer.transConv2D(input, kernel);

    printMat(output);
    cv::Mat expected = (cv::Mat_<float>(1, 4 * 4) << -1, 0, -2, 0,
                                                      0, 5, 0, 6,
                                                     -4, 0, -5, 0,
                                                      0, 8, 0, 9);
    EXPECT_TRUE(matEqual(output, expected));
}

TEST(Convolutional2DLayerTest, BackWard) {
    cv::Mat input = cv::Mat(3, 9, CV_32F);

    cv::randu(input, cv::Scalar::all(1), cv::Scalar::all(10));

    ConvolutionalLayer2D *convLayer;
    convLayer = new ConvolutionalLayer2D(3, 3, 2, 2, 1, 1, 1, 3, 6); 
    // 调用forward方法
    convLayer->forward(input);

    cv::Mat d0 = cv::Mat::ones(convLayer->getY().size(), CV_32F);
    convLayer->backward(d0);

    // printMat(convLayer->getDx());

    // cv::Mat expectedOutput = (cv::Mat_<float>(1,1) << some_value); 
    // EXPECT_EQ(convLayer->getY().at<cv::Mat>(0).at<float>(0, 0), expectedOutput.at<float>(0, 0));
}


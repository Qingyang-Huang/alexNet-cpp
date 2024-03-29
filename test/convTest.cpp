#include <gtest/gtest.h>
#include "layers/convolutionalLayer.h"
#include <opencv2/core/core.hpp>
#include "testUtils.h"

TEST(ConvolutionalLayer, Conv2D) {
    // Create an input matrix of size 3x3
    cv::Mat input = (cv::Mat_<float>(3, 3) << 1, 2, 3,
                                             4, 5, 6,
                                             7, 8, 9);
    // Define a kernel matrix of size 2x2
    cv::Mat kernel = (cv::Mat_<float>(2, 2) << 1, 0,
                                                0, -1);
    
    // Assuming ConvolutionalLayer has been properly initialized before this
    ConvolutionalLayer convLayer(3, 3, 2, 2, 1, 1, 1, 1, 1); // Adjust parameters as needed

    cv::Mat output = convLayer.conv2D(input, kernel);

    printMat(output);
    // Define the expected output matrix manually
    cv::Mat expected = (cv::Mat_<float>(2, 2) << 5, 5,
                                                   5, 5);

    // Check if the output matrix matches the expected matrix
    // Note: OpenCV doesn't have a direct way to compare matrices in GTest, so we use norm to check difference
    double diff = cv::norm(output, expected);
    EXPECT_NEAR(diff, 0.0, 1e-5);
}

TEST(ConvolutionalLayerTest, Forward) {
    cv::Mat input = (cv::Mat_<float>(7,7) << 1, 2, 3, 1, 2, 3, 1,
                                             4, 5, 6, 4, 5, 6, 4,
                                             1, 2, 3, 1, 2, 3, 1,
                                             4, 5, 6, 4, 5, 6, 4,
                                             1, 2, 3, 1, 2, 3, 1,
                                             4, 5, 6, 4, 5, 6, 4,
                                             1, 2, 3, 1, 2, 3, 1);
    ConvolutionalLayer *convLayer;
    convLayer = new ConvolutionalLayer(7, 7, 3, 3, 1, 1, 1, 1, 1); 

    

    // 调用forward方法
    convLayer->forward(input);

    printMat(convLayer->getY());

    // cv::Mat expectedOutput = (cv::Mat_<float>(1,1) << some_value); 
    // EXPECT_EQ(convLayer->getY().at<cv::Mat>(0).at<float>(0, 0), expectedOutput.at<float>(0, 0));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

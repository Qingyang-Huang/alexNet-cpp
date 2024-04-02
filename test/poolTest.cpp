#include <gtest/gtest.h>
#include "layers/poolingLayer2D.h"
#include <opencv2/core/core.hpp>
#include "testUtils.h"

TEST(PoolingLayer2D, MaxPooling) {
    cv::Mat input = (cv::Mat_<float>(1, 4*4) << 1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12,
                                             13, 14, 15, 16);
 
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 0, 1, 1, MAXPOOL); 

    poolLayer.forward(input);

   
    cv::Mat expected = (cv::Mat_<float>(1, 2*2) << 6, 8,
                                                  14, 16);
    EXPECT_TRUE(matEqual(poolLayer.getY(), expected));
}

TEST(PoolingLayer2D, MaxPooling2) {
    cv::Mat input = (cv::Mat_<float>(1, 4*4) << 1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12,
                                             13, 14, 15, 16);
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 1, MAXPOOL); 

    poolLayer.forward(input);

   
    cv::Mat expected = (cv::Mat_<float>(1, 3*3) << 1, 3, 4, 9, 11, 12, 13, 15, 16);

    // printMat(poolLayer.getY());

    EXPECT_TRUE(matEqual(poolLayer.getY(), expected));
}

TEST(PoolingLayer2D, MaxPooling3) {
    cv::Mat input = (cv::Mat_<float>(1, 3*3) << 1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9);
    PoolingLayer2D poolLayer(3, 3, 2, 2, 2, 2, 1, 1, 1, MAXPOOL); 

    poolLayer.forward(input);

    // printMat(poolLayer.getY());
   
    cv::Mat expected = (cv::Mat_<float>(1, 2*2) << 1, 3, 7, 9);
    EXPECT_TRUE(matEqual(poolLayer.getY(), expected));
}

TEST(PoolingLayer2D, MaxPooling4) {
    cv::Mat input = cv::Mat(3, 16, CV_32F);

    cv::randu(input, cv::Scalar::all(1), cv::Scalar::all(10));
    
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 3, 3, MAXPOOL);

    poolLayer.forward(input);

    // printMat(input);
    // printMat(poolLayer.getY());
    // printMat(poolLayer.getMaxPosition());


}

TEST(PoolingLayer2D, AvgPooling) {
    cv::Mat input = (cv::Mat_<float>(1, 4*4) << 1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12,
                                             13, 14, 15, 16);
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 0, 1, 1, AVGPOOL); 

    poolLayer.forward(input);

    // printMat(poolLayer.getY());
   
    cv::Mat expected = (cv::Mat_<float>(1, 2*2) << 3.5, 5.5,
                                                  11.5, 13.5);
    EXPECT_TRUE(matEqual(poolLayer.getY(), expected));

}

TEST(PoolingLayer2D, AvgPooling2) {
    cv::Mat input = cv::Mat(3, 16, CV_32F);

    cv::randu(input, cv::Scalar::all(1), cv::Scalar::all(10));
    
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 3, 3, AVGPOOL);

    poolLayer.forward(input);

    // printMat(input);
    // printMat(poolLayer.getY());


}

TEST(PoolingLayer2D, Back_Max) {
    cv::Mat input_f = (cv::Mat_<float>(1, 4*4) << 1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12,
                                             13, 14, 15, 16);
    cv::Mat input_b = (cv::Mat_<float>(1, 3*3) << 1, 3, 4, 9, 11, 12, 13, 15, 16);

    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 1, MAXPOOL); 

    poolLayer.forward(input_f);
    poolLayer.backward(input_b);

    // printMat(poolLayer.getDx());
    // printMat(poolLayer.getMaxPosition());
    cv::Mat expected = (cv::Mat_<float>(1, 4*4) << 1, 0, 3, 4, 
                                                   0, 0, 0, 0, 
                                                   9, 0, 11, 12, 
                                                   13, 0, 15, 16);
    EXPECT_TRUE(matEqual(poolLayer.getDx(), expected));
}

TEST(PoolingLayer2D, Back_Max2) {
    cv::Mat input = cv::Mat(3, 16, CV_32F);

    cv::randu(input, cv::Scalar::all(1), cv::Scalar::all(10));
    
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 3, 3, MAXPOOL);

    poolLayer.forward(input);

    cv::Mat d0 = cv::Mat(3, 9, CV_32F);
    cv::randu(d0, cv::Scalar::all(1), cv::Scalar::all(10));

    poolLayer.backward(d0);
}

TEST(PoolingLayer2D, Back_Avg) {
    cv::Mat input_f = (cv::Mat_<float>(1, 4*4) << 1, 2, 3, 4,
                                             5, 6, 7, 8,
                                             9, 10, 11, 12,
                                             13, 14, 15, 16);
    cv::Mat input_b = (cv::Mat_<float>(1, 3*3) << 1, 3, 4, 9, 11, 12, 13, 15, 16);

    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 1, AVGPOOL); 

    poolLayer.forward(input_f);
    poolLayer.backward(input_b);

    // printMat(poolLayer.getDx());
    // printMat(poolLayer.getMaxPosition());
    cv::Mat expected = (cv::Mat_<float>(1, 4*4) << 0.25, 0.75, 0.75, 1, 
                                                   2.25, 2.75, 2.75, 3, 
                                                   2.25, 2.75, 2.75, 3, 
                                                   3.25, 3.75, 3.75, 4);
    EXPECT_TRUE(matEqual(poolLayer.getDx(), expected));
}

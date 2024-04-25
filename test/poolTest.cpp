#include <gtest/gtest.h>
#include "layers/poolingLayer2D.h"


TEST(PoolingLayer2D, MaxPooling) {
    Tensor<float> input (1, 4*4, { 1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16});
 
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 1, 1, MAXPOOL); 

    poolLayer.forward(input);

   
    Tensor<float> expected (1, 3*3,{1, 3, 4, 
                                    9, 11, 12, 
                                    13, 15, 16});

    Tensor<size_t> expectedMaxPosition (1, 3*3*2, {0, 0, 
                                                    0, 2, 
                                                    0, 3, 
                                                    2, 0, 
                                                    2, 2, 
                                                    2, 3, 
                                                    3, 0, 
                                                    3, 2, 
                                                    3, 3});

    // poolLayer.getZ().reshape(1,3,3).print();
    // poolLayer.getMaxPosition().reshape(1,9,2).print();
    EXPECT_EQ(poolLayer.getZ(), expected);
    EXPECT_EQ(poolLayer.getMaxPosition(), expectedMaxPosition);
}

TEST(PoolingLayer2D, MaxPooling2) {
    Tensor<float> input (3, 4*4, { 1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16,
                                    1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16,
                                    1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16});
 
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 3, 3, MAXPOOL); 

    poolLayer.forward(input);

   
    Tensor<float> expected (3, 3*3,{ 1, 3, 4, 
                                    9, 11, 12, 
                                    13, 15, 16,
                                    1, 3, 4, 
                                    9, 11, 12, 
                                    13, 15, 16,
                                    1, 3, 4, 
                                    9, 11, 12, 
                                    13, 15, 16});

    Tensor<size_t> expectedMaxPosition (3, 3*3*2, {0, 0, 0, 2, 0, 3, 
                                                  2, 0, 2, 2, 2, 3, 
                                                  3, 0, 3, 2, 3, 3,
                                                  0, 0, 0, 2, 0, 3, 
                                                  2, 0, 2, 2, 2, 3, 
                                                  3, 0, 3, 2, 3, 3,
                                                  0, 0, 0, 2, 0, 3, 
                                                  2, 0, 2, 2, 2, 3, 
                                                  3, 0, 3, 2, 3, 3});

    // poolLayer.getZ().reshape(3,3,3).print();
    // poolLayer.getMaxPosition().reshape(3,9,2).print();
    EXPECT_EQ(poolLayer.getZ(), expected);
    EXPECT_EQ(poolLayer.getMaxPosition(), expectedMaxPosition);
}



TEST(PoolingLayer2D, AvgPooling) {
    Tensor<float> input (1, 4*4, { 1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16});
 
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 1, 1, AVGPOOL); 

    poolLayer.forward(input);

   
    Tensor<float> expected (1, 3*3,{0.25, 1.25, 1, 
                                    3.5, 8.5, 5, 
                                    3.25, 7.25, 4});

    // poolLayer.getZ().reshape(1,3,3).print();
    EXPECT_EQ(poolLayer.getZ(), expected);
}

TEST(PoolingLayer2D, AvgPooling2) {
    Tensor<float> input (3, 4*4, { 1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16,
                                    1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16,
                                    1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16});
 
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 3, 3, AVGPOOL); 

    poolLayer.forward(input);

   
    Tensor<float> expected (3, 3*3,{0.25, 1.25, 1, 
                                    3.5, 8.5, 5, 
                                    3.25, 7.25, 4,
                                    0.25, 1.25, 1, 
                                    3.5, 8.5, 5, 
                                    3.25, 7.25, 4,
                                    0.25, 1.25, 1, 
                                    3.5, 8.5, 5, 
                                    3.25, 7.25, 4});

    // poolLayer.getZ().reshape(3,3,3).print();
    EXPECT_EQ(poolLayer.getZ(), expected);
}

TEST(PoolingLayer2D, MaxPoolingBack) {
    Tensor<float> input (1, 4*4, { 1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16});
 
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 1, 1, MAXPOOL); 

    poolLayer.forward(input);

    Tensor<float> d0 (1, 3*3, {1, 2, 3, 
                                4, 5, 6, 
                                7, 8, 9});
    
    poolLayer.backward(d0);

   
    Tensor<float> expected (1, 4*4,{1, 0, 2, 3, 
                                    0, 0, 0, 0, 
                                    4, 0, 5, 6, 
                                    7, 0, 8, 9,});

    // poolLayer.getDx().reshape(1,4,4).print();
    EXPECT_EQ(poolLayer.getDx(), expected);
}

TEST(PoolingLayer2D, MaxPoolingBack2) {
    Tensor<float> input (3, 4*4, { 1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16,
                                    1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16,
                                    1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16});
 
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 3, 3, MAXPOOL); 

    poolLayer.forward(input);

    Tensor<float> d0 (3, 3*3, {1, 2, 3, 
                                4, 5, 6, 
                                7, 8, 9,
                                1, 2, 3, 
                                4, 5, 6, 
                                7, 8, 9,
                                1, 2, 3, 
                                4, 5, 6, 
                                7, 8, 9});
    
    poolLayer.backward(d0);

    Tensor<float> expected(3, 4*4,{1, 0, 2, 3, 
                                    0, 0, 0, 0, 
                                    4, 0, 5, 6, 
                                    7, 0, 8, 9,
                                    1, 0, 2, 3, 
                                    0, 0, 0, 0, 
                                    4, 0, 5, 6, 
                                    7, 0, 8, 9,
                                    1, 0, 2, 3, 
                                    0, 0, 0, 0, 
                                    4, 0, 5, 6, 
                                    7, 0, 8, 9,});

    EXPECT_EQ(poolLayer.getDx(), expected);
}


TEST(PoolingLayer2D, AVGPoolingBack) {
    Tensor<float> input (1, 4*4, { 1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16});
 
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 1, 1, AVGPOOL); 

    poolLayer.forward(input);

    Tensor<float> d0 (1, 3*3, {1, 2, 3, 
                                4, 5, 6, 
                                7, 8, 9});
    
    poolLayer.backward(d0);

   
    Tensor<float> expected (1, 4*4,{0.25, 0.5, 0.5, 0.75,
                                    1, 1.25, 1.25, 1.5, 
                                    1, 1.25, 1.25, 1.5,
                                    1.75, 2, 2, 2.25});

    // poolLayer.getDx().reshape(1,4,4).print();
    EXPECT_EQ(poolLayer.getDx(), expected);
}

TEST(PoolingLayer2D, AvgPoolingBack2) {
    Tensor<float> input (3, 4*4, { 1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16,
                                    1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16,
                                    1, 2, 3, 4,
                                    5, 6, 7, 8,
                                    9, 10, 11, 12,
                                    13, 14, 15, 16});
 
    PoolingLayer2D poolLayer(4, 4, 2, 2, 2, 2, 1, 1, 3, 3, AVGPOOL); 

    poolLayer.forward(input);

   
    Tensor<float> d0 (3, 3*3, {1, 2, 3, 
                                4, 5, 6, 
                                7, 8, 9,
                                1, 2, 3, 
                                4, 5, 6, 
                                7, 8, 9,
                                1, 2, 3, 
                                4, 5, 6, 
                                7, 8, 9});
    
    poolLayer.backward(d0);

   
    Tensor<float> expected (3, 4*4,{0.25, 0.5, 0.5, 0.75,
                                    1, 1.25, 1.25, 1.5, 
                                    1, 1.25, 1.25, 1.5,
                                    1.75, 2, 2, 2.25,
                                    0.25, 0.5, 0.5, 0.75,
                                    1, 1.25, 1.25, 1.5, 
                                    1, 1.25, 1.25, 1.5,
                                    1.75, 2, 2, 2.25,
                                    0.25, 0.5, 0.5, 0.75,
                                    1, 1.25, 1.25, 1.5, 
                                    1, 1.25, 1.25, 1.5,
                                    1.75, 2, 2, 2.25
                                });

    // poolLayer.getDx().reshape(3,4,4).print();
    EXPECT_EQ(poolLayer.getDx(), expected);
}
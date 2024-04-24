#include <gtest/gtest.h>
#include "layers/convolutionalLayer2D.h"

TEST(ConvolutionalLayer2D, Conv2D) {
    Tensor<float> input(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});

    Tensor<float> kernel(1, 2 * 2, {1, 0,
                                   0, -1});
    Tensor<float> bias(1, 1, {1});
    
    ConvolutionalLayer2D convLayer(3, 3, 2, 2, 1, 1, 1, 1, 1, 1, true); 

    convLayer.setKernels(kernel);

    convLayer.setBias(bias);

    convLayer.conv2D(input);

    Tensor<float> expected (1, 4 * 4, { 0, -1, -2, 1, -3, -3, -3, 4, -6, -3, -3, 7, 1, 8, 9, 10});

    EXPECT_EQ(convLayer.getZ(), expected);
}

TEST(ConvolutionalLayer2D, Conv2D2) {
    Tensor<float> input(3, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9,
                                  1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9,
                                  1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});

    Tensor<float> kernel(1, 3 * 2 * 2, {1, 0,
                                        0, -1,
                                        1, 0,
                                        0, -1,
                                        1, 0,
                                        0, -1});
    Tensor<float> bias(1, 1, {1});
    
    ConvolutionalLayer2D convLayer(3, 3, 2, 2, 1, 1, 1, 1, 3, 1, true); 

    convLayer.setKernels(kernel);

    convLayer.setBias(bias);

    convLayer.conv2D(input);

    convLayer.getZ().reshape(1, 1, 4 * 4).print();
    Tensor<float> expected (1, 4 * 4, {-2, -5, -8, 1, -11, -11, -11, 10, -20, -11, -11, 19, 1, 22, 25, 28});

    EXPECT_EQ(convLayer.getZ(), expected);
}

TEST(ConvolutionalLayer2D, Conv2D3) {
    Tensor<float> input(3, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9,
                                  1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9,
                                  1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});

    Tensor<float> kernel(3, 3 * 2 * 2, {1, 0,
                                    0, -1,
                                    1, 0,
                                    0, -1,
                                    1, 0,
                                    0, -1, 
                                    1, 0,
                                    0, -1,
                                    1, 0,
                                    0, -1,
                                    1, 0,
                                    0, -1,
                                    1, 0,
                                    0, -1,
                                    1, 0,
                                    0, -1,
                                    1, 0,
                                    0, -1});
    Tensor<float> bias(1, 3, {1, 2, 3});
    
    ConvolutionalLayer2D convLayer(3, 3, 2, 2, 1, 1, 1, 1, 3, 3, true); 

    convLayer.setKernels(kernel);

    convLayer.setBias(bias);

    convLayer.conv2D(input);

    convLayer.getZ().reshape(1, 3, 4 * 4).print();
    Tensor<float> expected (3, 4 * 4, {-2, -5, -8, 1, -11, -11, -11, 10, -20, -11, -11, 19, 1, 22, 25, 28, 
                                        -1, -4, -7, 2, -10, -10, -10, 11, -19, -10, -10, 20, 2, 23, 26, 29, 
                                        0, -3, -6, 3, -9, -9, -9, 12, -18, -9, -9, 21, 3, 24, 27, 30});

    EXPECT_EQ(convLayer.getZ(), expected);
}

TEST(ConvolutionalLayer2D, Conv2D_stride) {
    Tensor<float> input(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});

    Tensor<float> kernel(1, 2 * 2, {1, 0,
                                   0, -1});
    Tensor<float> bias(1, 1, {1});
    
    ConvolutionalLayer2D convLayer(3, 3, 2, 2, 2, 2, 2, 2, 1, 1, true); 

    convLayer.setKernels(kernel);

    convLayer.setBias(bias);

    convLayer.conv2D(input);

    Tensor<float> expected (1, 3 * 3, {1, 1, 1, 1, -3, 4, 1, 8, 10});

    EXPECT_EQ(convLayer.getZ(), expected);
}

TEST(Convolutional2DLayerTest, TransConv1) {
    Tensor<float> input(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});

    Tensor<float> kernel(1, 2 * 2, {1, 0,
                                   0, -1});
    Tensor<float> bias(1, 1, {1});

    ConvolutionalLayer2D convLayer(4, 4, 2, 2, 2, 2, 1, 1, 1, 1); 

    convLayer.setKernels(kernel);

    convLayer.setBias(bias);

    convLayer.transConv2D(input);

    convLayer.getDx().reshape(1, 1, 4 * 4).print();

    Tensor<float> expected (1, 4 * 4, {-1, 0, -2, 0,
                                        0, 5, 0, 6,
                                        -4, 0, -5, 0,
                                        0, 8, 0, 9});
    EXPECT_EQ(convLayer.getDx(), expected);

}

TEST(Convolutional2DLayerTest, TransConv2) {
    Tensor<float> input(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9,
                                  });

    Tensor<float> kernel(1, 3 * 2 * 2, {1, 0,
                                        0, -1,
                                        1, 0,
                                        0, -1,
                                        1, 0,
                                        0, -1});
    Tensor<float> bias(1, 1, {1});

    ConvolutionalLayer2D convLayer(4, 4, 2, 2, 2, 2, 1, 1, 3, 1); 

    convLayer.setKernels(kernel);

    convLayer.setBias(bias);

    convLayer.transConv2D(input);

    convLayer.getDx().reshape(1, 3, 4 * 4).print();

    Tensor<float> expected (3, 4 * 4, {-1, 0, -2, 0,
                                        0, 5, 0, 6,
                                        -4, 0, -5, 0,
                                        0, 8, 0, 9, -1, 0, -2, 0,
                                        0, 5, 0, 6,
                                        -4, 0, -5, 0,
                                        0, 8, 0, 9, -1, 0, -2, 0,
                                        0, 5, 0, 6,
                                        -4, 0, -5, 0,
                                        0, 8, 0, 9});
    EXPECT_EQ(convLayer.getDx(), expected);
}

TEST(Convolutional2DLayerTest, backward1) {
    Tensor<float> input(1, 4 * 4, {1, 2, 3, 1,
                                  4, 5, 6, 4,
                                  7, 8, 9, 7,
                                  4, 5, 6, 4});

    Tensor<float> kernel(1, 2 * 2, {1, 0,
                                   0, -1});
    Tensor<float> bias(1, 1, {1});

    ConvolutionalLayer2D convLayer(4, 4, 2, 2, 2, 2, 1, 1, 1, 1); 

    convLayer.setKernels(kernel);

    convLayer.setBias(bias);

    convLayer.forward(input);

    convLayer.getZ().reshape(1, 1, 3 * 3).print();


    Tensor<float> loss(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});

    convLayer.backward(loss);

    convLayer.getD().reshape(1, 1, 3 * 3).print();

    convLayer.getDx().reshape(1, 4, 4).print();

    Tensor<float> expected (1, 4 * 4, {0, 0, 0, 0, 
                                        0, 0, 0, 6, 
                                        0, 0, 0, 0, 
                                        0, 8, 0, 9});
    EXPECT_EQ(convLayer.getDx(), expected);
}

TEST(Convolutional2DLayerTest, backward2) {
    Tensor<float> input(3, 4 * 4, {1, 2, 3, 4,
                                  4, 5, 6, 7,
                                  7, 8, 9, 10,
                                  1, 2, 3, 4, 
                                  1, 2, 3, 4,
                                  4, 5, 6, 7,
                                  7, 8, 9, 10,
                                  1, 2, 3, 4, 
                                  1, 2, 3, 4,
                                  4, 5, 6, 7,
                                  7, 8, 9, 10,
                                  1, 2, 3, 4,
                                  });

    Tensor<float> kernel(1, 3 * 2 * 2, {1, 0,
                                        0, -1,
                                        1, 0,
                                        0, -1,
                                        1, 0,
                                        0, -1});
    Tensor<float> bias(1, 1, {1});

    ConvolutionalLayer2D convLayer(4, 4, 2, 2, 2, 2, 1, 1, 3, 1); 

    convLayer.setKernels(kernel);

    convLayer.setBias(bias);

    convLayer.forward(input);

    convLayer.getZ().reshape(1, 3, 3).print();

    Tensor<float> loss(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});

    convLayer.backward(loss);

    // convLayer.getD().reshape(1, 3, 3).print();

    // convLayer.getDx().reshape(3, 4, 4).print();

    Tensor<float> expected (3, 4 * 4, {0, 0, 0, 0, 
                                        0, 0, 0, 6, 
                                        0, 0, 0, 0, 
                                        0, 8, 0, 9, 
                                        0, 0, 0, 0, 
                                        0, 0, 0, 6, 
                                        0, 0, 0, 0, 
                                        0, 8, 0, 9, 
                                        0, 0, 0, 0, 
                                        0, 0, 0, 6, 
                                        0, 0, 0, 0, 
                                        0, 8, 0, 9});
    EXPECT_EQ(convLayer.getDx(), expected);
}

TEST(Convolutional2DLayerTest, update1) {
    Tensor<float> input(1, 4 * 4, {1, 2, 3, 1,
                                  4, 5, 6, 4,
                                  7, 8, 9, 7,
                                  4, 5, 6, 4});

    Tensor<float> kernel(1, 2 * 2, {1, 0, 0, -1});
    Tensor<float> bias(1, 1, {1});

    ConvolutionalLayer2D convLayer(4, 4, 2, 2, 2, 2, 1, 1, 1, 1, true); 

    convLayer.setKernels(kernel);

    convLayer.setBias(bias);

    convLayer.forward(input);

    convLayer.getZ().reshape(1, 1, 3 * 3).print();


    Tensor<float> loss(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});

    convLayer.backward(loss);

    Tensor<float> al(1, 4 * 4, {1, 2, 3, 1,
                                  4, 5, 6, 4,
                                  7, 8, 9, 7,
                                  4, 5, 6, 4});
    
    convLayer.updateWeight(al, 0.1);

    convLayer.getKernels().print();

    convLayer.getBias().print();

    Tensor<float> expected (1, 2 * 2, {-9, 
                                        -7.6, 
                                        -4.5, 
                                        -1, });

    Tensor<float> expectBias (1, 1, {-2.3});
    EXPECT_EQ(convLayer.getKernels(), expected);
    EXPECT_EQ(convLayer.getBias(),expectBias);
}

TEST(Convolutional2DLayerTest, update2) {
    Tensor<float> input(3, 4 * 4, {1, 2, 3, 4,
                                  4, 5, 6, 7,
                                  7, 8, 9, 10,
                                  1, 2, 3, 4, 
                                  1, 2, 3, 4,
                                  4, 5, 6, 7,
                                  7, 8, 9, 10,
                                  1, 2, 3, 4, 
                                  1, 2, 3, 4,
                                  4, 5, 6, 7,
                                  7, 8, 9, 10,
                                  1, 2, 3, 4,
                                  });

    Tensor<float> kernel(1, 3 * 2 * 2, {1, 0,
                                        0, -1,
                                        1, 0,
                                        0, -1,
                                        1, 0,
                                        0, -1});
    Tensor<float> bias(1, 1, {1});

    ConvolutionalLayer2D convLayer(4, 4, 2, 2, 2, 2, 1, 1, 3, 1, true); 

    convLayer.setKernels(kernel);

    convLayer.setBias(bias);

    convLayer.forward(input);

    Tensor<float> loss(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});

    convLayer.backward(loss);

    Tensor<float> al(3, 4 * 4, {1, 2, 3, 4,
                                  4, 5, 6, 7,
                                  7, 8, 9, 10,
                                  1, 2, 3, 4, 
                                  1, 2, 3, 4,
                                  4, 5, 6, 7,
                                  7, 8, 9, 10,
                                  1, 2, 3, 4, 
                                  1, 2, 3, 4,
                                  4, 5, 6, 7,
                                  7, 8, 9, 10,
                                  1, 2, 3, 4,
                                  });
    convLayer.getD().print();

    convLayer.updateWeight(al, 0.1);

    convLayer.getKernels().print();

    Tensor<float> expected (1, 3 * 2 * 2, {-8.4, -3.1, 
                                            -7.2, -1, 
                                            -8.4, -3.1, 
                                            -7.2, -1, 
                                            -8.4, -3.1, 
                                            -7.2, -1, 
                                        });

    EXPECT_EQ(convLayer.getKernels(), expected);
}
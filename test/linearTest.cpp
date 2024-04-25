#include <gtest/gtest.h>
#include "layers/linearLayer.h"
#include "layers/dropout.h"
#include "layers/activation.h"

TEST (LinearLayer, forward) {
    LinearLayer layer(9, 4);
    Tensor<float> input(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});
    layer.setWeight(Tensor<float>(4, 9, {1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 2,
                                         3, 3, 3, 3, 3, 3, 3, 3, 3,
                                         4, 4, 4, 4, 4, 4, 4, 4, 4}));
    layer.setBias(Tensor<float>(1, 4, {1, 2, 3, 4}));
    layer.forward(input);
    Tensor<float> expected(1, 4, {46, 92, 138, 184});
    layer.getZ().print();
    EXPECT_EQ(layer.getZ(), expected);
}

TEST (LinearLayer, backward) {
    LinearLayer layer(9, 4);
    Tensor<float> input(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});
    layer.setWeight(Tensor<float>(4, 9, {1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 2,
                                         3, 3, 3, 3, 3, 3, 3, 3, 3,
                                         4, 4, 4, 4, 4, 4, 4, 4, 4}));
    layer.setBias(Tensor<float>(1, 4, {1, 2, 3, 4}));
    layer.forward(input);

    Tensor<float> d0(1, 4, {1, 2, 3, 4});
    layer.backward(d0);
    Tensor<float> expected(1, 9, {30, 30, 30, 30, 30, 30, 30, 30, 30});
    layer.getDx().print();
    layer.updateWeight(input, 0.1);
    layer.getWData().reshape(1,4,9).print();
    Tensor<float> expectedWeight(4, 9, {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 
                                        1.8, 1.6, 1.4, 1.2, 1, 0.8, 0.6, 0.4, 0.2, 
                                        2.7, 2.4, 2.1, 1.8, 1.5, 1.2, 0.9, 0.6, 0.3, 
                                        3.6, 3.2, 2.8, 2.4, 2, 1.6, 1.2, 0.8, 0.4, });
    EXPECT_EQ(layer.getDx(), expected);
    EXPECT_EQ(layer.getWData(), expectedWeight);
}

TEST (LinearLayer, DropOut) {
    Tensor<float> input(1, 3 * 3, {1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9});
    dropout(input).reshape(1,3,3).print();
}

TEST (LinearLayer, SoftMax){
    Tensor<float> input(1, 4, {1, 2, 3, 4});
    Tensor<float> output = softmax(input).reshape(1,1,4);
    softmaxDerivative(output).reshape(1,1,4).print();
//     Tensor<float> expected(1, 4, {0.0320586, 0.0871443, 0.2368828, 0.6439143});
//     Tensor<float> expectedD(1, 4, {0.1966119, 0.1049936, 0.0706507, 0.0529881});
//     EXPECT_EQ(softmax(input), expected);
//     EXPECT_EQ(softmaxDerivative(input), expectedD);
}

TEST (LinearLayer, Relu){
    Tensor<float> input(1, 4, {1, -1, 3, -5});
    Tensor<float> output(1,4);
     Tensor<float> outputD(1,4);
    for(int i = 0; i < input.size(); i++){
        output(i) = relu(input(i));
        outputD(i) = reluDerivative(input(i));
    }
    Tensor<float> expected(1, 4, {1, 0, 3, 0});
    Tensor<float> expectedD(1, 4, {1, 0, 1, 0});
    EXPECT_EQ(output, expected);
    EXPECT_EQ(outputD, expectedD);
    
}
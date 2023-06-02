#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/maxpooling_op.hpp"
#include "layer/maxpooling_layer.hpp"
#include "factory/layer_factory.hpp"


TEST(test_layer, forward_maxpooling) {
  using namespace kuiper_infer;
  
    Shape kernel_size = {2, 2};
    Shape stride = {2, 2};
    Shape padding = {1, 1};


    std::shared_ptr<Operator> maxpooling_op = std::make_shared<MaxPoolingOp>(kernel_size, stride, padding);
    std::shared_ptr<Layer> maxpooling_layer = LayerRegister::CreateLayer(maxpooling_op);

    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 4, 4);

    arma::fmat inp = {{1, 2, 3, 4}, 
                    {5, 6, 7, 8}, 
                    {9, 10, 11, 12}, 
                    {13, 14, 15, 16}};

    input->slice(0) = inp;

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    inputs.push_back(input);

    maxpooling_layer->Forward(inputs, outputs);
    
    ASSERT_EQ(outputs.size(), 1);
    const auto& output = outputs.at(0);
    
// {{1, 3, 4},
//  {10, 11, 12},
//  {13, 15, 16},}
    
    ASSERT_EQ(output->rows(), 3);
    ASSERT_EQ(output->cols(), 3);
    ASSERT_EQ(output->channels(), 1);

    ASSERT_EQ(output->at(0, 0, 0), 1);
    ASSERT_EQ(output->at(0, 0, 1), 3);
    ASSERT_EQ(output->at(0, 0, 2), 4);
    ASSERT_EQ(output->at(0, 1, 0), 9);
    ASSERT_EQ(output->at(0, 1, 1), 11);
    ASSERT_EQ(output->at(0, 1, 2), 12);
    ASSERT_EQ(output->at(0, 2, 0), 13);
    ASSERT_EQ(output->at(0, 2, 1), 15);
    ASSERT_EQ(output->at(0, 2, 2), 16);
    std::cout << input->data();
    std::cout << output->data();
}
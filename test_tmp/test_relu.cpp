#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"


// 没有注册表
TEST(test_relu, forward_relu1) {
    using namespace kuiper_infer;
    float threshold = 0.f;

    std::shared_ptr<Operator> relu_op = std::make_shared<ReLUOperator>(threshold);

    std::shared_ptr<Tensor<float>> input_data = std::make_shared<Tensor<float>>(1, 1, 3);
    // (-1, -2, 3)
    input_data->index(0) = -1.f; //output对应的应该是0
    input_data->index(1) = -2.f; //output对应的应该是0
    input_data->index(2) = 3.f; //output对应的应该是3

    // 一个大小为1的batch
    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    
    std::vector<std::shared_ptr<Tensor<float>>> outputs;

    inputs.push_back(input_data);

    ReLULayer layer(relu_op);

    layer.Forward(inputs, outputs);
    ASSERT_EQ(outputs.size(), 1);
                                     
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_EQ(outputs.at(i)->index(0), 0.f);
        ASSERT_EQ(outputs.at(i)->index(1), 0.f);
        ASSERT_EQ(outputs.at(i)->index(2), 3.f);
        }
}

TEST(test_layer, forward_relu2) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  std::shared_ptr<Operator> relu_op = std::make_shared<ReLUOperator>(thresh);
  std::shared_ptr<Layer> relu_layer = LayerRegister::CreateLayer(relu_op);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);
  relu_layer->Forward(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}
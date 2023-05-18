#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/sigmoid_op.hpp"
#include "layer/sigmoid_layer.hpp"
#include "factory/layer_factory.hpp"
#include <limits>

TEST(test_layer, forward_sigmoid) {
    using namespace kuiper_infer;
    
    std::shared_ptr<Operator> sigmoid_op = std::make_shared<SigmoidOperator>();
    std::shared_ptr<Layer> sigmoid_layer = LayerRegister::CreateLayer(sigmoid_op);

    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);

    float min_float = std::numeric_limits<float>::lowest();
    float max_float = std::numeric_limits<float>::max();

    input->index(0) = min_float;
    input->index(1) = 0.f;
    input->index(2) = max_float;

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    std::vector<std::shared_ptr<Tensor<float>>> outputs;
    
    inputs.push_back(input);
    sigmoid_layer->Forward(inputs, outputs);
    
    ASSERT_EQ(outputs.size(), 1);
    
    for (int i = 0; i < outputs.size(); ++i) {
        ASSERT_NEAR(outputs.at(i)->index(0), 0.f, 1e-6);
        ASSERT_NEAR(outputs.at(i)->index(1), 0.5f, 1e-6);
        ASSERT_NEAR(outputs.at(i)->index(2), 1.f, 1e-6);
    }
}
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "ops/op.hpp"
#include "layer/conv_layer.hpp"
#include "data/tensor_util.hpp"

// 单卷积单通道
TEST(test_layer, conv1) {
    using namespace kuiper_infer;
    LOG(INFO) << "My convolution test!";
    
    Shape stride = {1, 1};
    Shape padding = {0, 0};
    
    ConvOp *conv_op = new ConvOp(stride, padding, false, 1);
    // 单个卷积核的情况
    std::vector<float> values;
    for (int i = 0; i < 3; ++i) {
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
        values.push_back(float(i + 1));
    }
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(1, 3, 3);
    weight1->Fill(values);
    LOG(INFO) << "weight:";
    weight1->Show();
    // 设置权重
    std::vector<sftensor> weights;
    weights.push_back(weight1);
    conv_op->set_weights(weights);
    std::shared_ptr<Operator> op = std::shared_ptr<ConvOp>(conv_op);

    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat input_data = {{1, 2, 3, 4}, 
                            {5, 6, 7, 8}, 
                            {9, 10, 11, 12}, 
                            {13, 14, 15, 16}};
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(1, 4, 4);
    input->slice(0) = input_data;
    LOG(INFO) << "input:";
    input->Show();
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    ConvLayer layer(op);
    std::vector<std::shared_ptr<ftensor >> outputs(1);

    layer.Forward(inputs, outputs);
    LOG(INFO) << "result: ";
    for (int i = 0; i < outputs.size(); ++i) {
        outputs.at(i)->Show();
    }
}

// 多卷积多通道
TEST(test_layer, conv2) {
    using namespace kuiper_infer;
    
    LOG(INFO) << "My convolution test!";

    Shape stride = {1, 1};
    Shape padding = {0, 0};
    
    ConvOp *conv_op = new ConvOp(stride, padding, false, 1);
    // 单个卷积核的情况
    std::vector<float> values;

    arma::fmat weight_data = {{1 ,1, 1},
                            {2 ,2, 2},
                            {3 ,3, 3}};
    // 初始化三个卷积核
    std::shared_ptr<ftensor> weight1 = std::make_shared<ftensor>(3, 3, 3);
    weight1->slice(0) = weight_data;
    weight1->slice(1) = weight_data;
    weight1->slice(2) = weight_data;

    std::shared_ptr<ftensor> weight2 = TensorClone(weight1);
    std::shared_ptr<ftensor> weight3 = TensorClone(weight1);

    LOG(INFO) << "weight:";
    weight1->Show();
    // 设置权重
    std::vector<sftensor> weights;
    weights.push_back(weight1);
    weights.push_back(weight2);
    weights.push_back(weight3);

    conv_op->set_weights(weights);
    std::shared_ptr<Operator> op = std::shared_ptr<ConvOp>(conv_op);

    std::vector<std::shared_ptr<ftensor >> inputs;
    arma::fmat inp = {{1, 2, 3, 4}, 
                    {5, 6, 7, 8}, 
                    {9, 10, 11, 12}, 
                    {13, 14, 15, 16}};
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 4, 4);
    input->slice(0) = inp;
    input->slice(1) = inp;
    input->slice(2) = inp;

    LOG(INFO) << "input:";
    input->Show();
    // 权重数据和输入数据准备完毕
    inputs.push_back(input);
    ConvLayer layer(op);
    std::vector<std::shared_ptr<ftensor >> outputs(1);

    layer.Forward(inputs, outputs);
    LOG(INFO) << "result: ";
    for (int i = 0; i < outputs.size(); ++i) {
        outputs.at(i)->Show();
    }
}
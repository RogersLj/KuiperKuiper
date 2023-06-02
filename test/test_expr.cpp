#include "runtime/runtime_ir.hpp"
#include <gtest/gtest.h>
#include <gtest/gtest.h>

TEST(test_expr, expr_layer) {
    using namespace kuiper_infer;
    const std::string &param_path = "../tmp/expr_layer.pnnx.param";
    const std::string &weight_path = "../tmp/expr_layer.pnnx.bin";
    RuntimeGraph graph(param_path, weight_path);
    graph.Build("pnnx_input_0", "pnnx_output_0");
    const auto &operators = graph.operators();
    LOG(INFO) << "operator size: " << operators.size();
    uint32_t batch_size = 1;
    
    arma::fmat input_data = {{1, 2, 3, 4}, 
                                {5, 6, 7, 8}, 
                                {9, 10, 11, 12}, 
                                {13, 14, 15, 16}};

    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 4, 4);

    for (uint32_t i = 0; i < 3; i ++) {
        input->slice(i) = input_data;
    }
  
    input->Show();
  
    std::vector<sftensor> inputs(batch_size);
    for (uint32_t i = 0; i < batch_size; ++i) {
        inputs.at(i) = input;
    }
    const std::vector<sftensor> &outputs = graph.Forward(inputs, true);

    LOG(INFO) << "----------------------输出的形状：---------\n" << outputs.size() << ", " << outputs.at(0)->shape().at(0) << ", " << outputs.at(0)->shape().at(1) << ", " << outputs.at(0)->shape().at(2);

    for (uint32_t i = 0; i < batch_size; ++i) {
        outputs.at(i)->Show();
    }
}
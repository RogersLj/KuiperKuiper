#include <glog/logging.h>
#include <gtest/gtest.h>
#include "runtime/runtime_ir.hpp"

TEST(test_initinoutput, init_init_input) {
  using namespace kuiper_infer;
  const std::string &param_path = "../tmp/ten.pnnx.param";
  const std::string &bin_path = "../tmp/ten.pnnx.bin";
  RuntimeGraph graph(param_path, bin_path);
  graph.Init();
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &operators = graph.operators();
  for (const auto &operator_ : operators) {
    LOG(INFO) << "type: " << operator_->type << " name: " << operator_->name;
    const std::map<std::string, std::shared_ptr<RuntimeOperand>> &
        input_operands_map = operator_->input_operands;
    for (const auto &input_operand : input_operands_map) {
      std::string shape_str;
      for (const auto &dim : input_operand.second->shape) {
        shape_str += std::to_string(dim) + " ";
      }
      LOG(INFO) << "operand name: " << input_operand.first << " operand shape: " << shape_str;
    }
    LOG(INFO) << "-------------------------------------------------------";
  }
  // 获取输入空间初始化后的Operator
  const auto &operators2 = graph.operators();
  const auto
      &operators3 = std::vector<std::shared_ptr<RuntimeOperator>>(operators2.begin() + 1, operators2.begin() + 5);

  // 获取name为conv1 relu1的2个op进行校验

  // 校验conv1
  // operand name: pnnx_input_0 operand shape: 2 3 128 128 所以我们在下方要求size=2（也就是batch等于2） 通道c=3 rows = 128 cols = 128
  const auto &conv1 = *operators3.begin();
  const auto &conv1_input_operand = conv1->input_operands;
  ASSERT_EQ(conv1_input_operand.find("pnnx_input_0")->second->datas.size(), 2);
  const std::vector<std::shared_ptr<Tensor<float>>>
      &datas_conv1 = conv1_input_operand.at("pnnx_input_0")->datas; // datas是被准备好的空间,实际上的大小
  for (const auto &data_conv1 : datas_conv1) {
    ASSERT_EQ(data_conv1->shape().at(0), 3);
    ASSERT_EQ(data_conv1->shape().at(1), 128);
    ASSERT_EQ(data_conv1->shape().at(2), 128);
  }

  // operand name: conv1 operand shape: 2 64 128 128
  const auto &relu1 = *(operators3.begin() + 1);
  const auto &relu1_input_operand = relu1->input_operands;
  ASSERT_EQ(relu1_input_operand.find("conv1")->second->datas.size(), 2);
  const std::vector<std::shared_ptr<Tensor<float>>> &datas_relu1 = relu1_input_operand.at("conv1")->datas;
  for (const auto &data_relu1 : datas_relu1) {
    ASSERT_EQ(data_relu1->shape().at(0), 64);
    ASSERT_EQ(data_relu1->shape().at(1), 128);
    ASSERT_EQ(data_relu1->shape().at(2), 128);
  }
}


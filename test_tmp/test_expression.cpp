#include "parser/parse_expression.hpp"
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "parser/parse_expression.hpp"

#include "layer/expression_layer.hpp"
#include "ops/expression_op.hpp"



TEST(test_expression, expression1) {
  using namespace kuiper_infer;
  const std::string &statement = "add(mul(@0,@1),@2)";
  ExpressionParser parser(statement);
  const auto &node_tokens = parser.Generate();
  ExpressionParser::PrintNodes(node_tokens.back());
}

TEST(test_expression, complex) {
  using namespace kuiper_infer;
  const std::string &expression = "add(mul(@0,@1),@2)";
  std::shared_ptr<ExpressionOp> expression_op = std::make_shared<ExpressionOp>(expression);
  ExpressionLayer layer(expression_op);
  std::vector<std::shared_ptr<ftensor >> inputs;
  std::vector<std::shared_ptr<ftensor >> outputs;

  int batch_size = 4;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(1.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(2.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> input = std::make_shared<ftensor>(3, 224, 224);
    input->Fill(3.f);
    inputs.push_back(input);
  }

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<ftensor> output = std::make_shared<ftensor>(3, 224, 224);
    outputs.push_back(output);
  }
  layer.Forward(inputs, outputs);
  for (int i = 0; i < batch_size; ++i) {
    const auto &result = outputs.at(i);
    for (int j = 0; j < result->size(); ++j) {
      ASSERT_EQ(result->index(j), 5.f);
    }
  }
}
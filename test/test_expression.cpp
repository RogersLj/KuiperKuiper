#include "parser/parse_expression.hpp"
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "parser/parse_expression.hpp"



TEST(test_expression, expression1) {
  using namespace kuiper_infer;
  const std::string &statement = "add(mul(@0,@1),@2)";
  ExpressionParser parser(statement);
  const auto &node_tokens = parser.Generate();
  ExpressionParser::PrintNodes(node_tokens.back());
}
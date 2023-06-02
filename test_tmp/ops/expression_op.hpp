#ifndef KUIPER_INFER_OPS_EXPRESSION_OP_HPP
#define KUIPER_INFER_OPS_EXPRESSION_OP_HPP

#include <vector>
#include <string>
#include <memory>
#include "op.hpp"
#include "parser/parse_expression.hpp"


namespace kuiper_infer {

class ExpressionOp : public Operator {

public:

    explicit ExpressionOp(const std::string &expression);

    std::vector<std::shared_ptr<TokenNode>> Generate(); // 可以返回计算图
    // expression layer 不是一个运算，而是多个运算


private:
    std::string expression_; // 表达式
    std::vector<std::shared_ptr<TokenNode>> nodes_; // 左右根存储的节点,只有数字和add mul
    std::shared_ptr<ExpressionParser> parser_; // 用于构建计算图的解析器

};

}

#endif
#include <glog/logging.h>
#include "ops/expression_op.hpp"

namespace kuiper_infer {
    
ExpressionOp::ExpressionOp(const std::string& expression) : Operator(OpType::kOperatorExpression), expression_(expression) {
    this->parser_ = std::make_shared<ExpressionParser>(expression);
}

std::vector<std::shared_ptr<TokenNode>> ExpressionOp::Generate() {
    CHECK(this->parser_ != nullptr);
    this->nodes_ = this->parser_->Generate();
    return this->nodes_;
}

}
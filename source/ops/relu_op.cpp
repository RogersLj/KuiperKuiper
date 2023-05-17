#include "ops/relu_op.hpp"

namespace kuiper_infer {
    
ReLUOperator::ReLUOperator(float threshold) : threshold_(threshold), Operator(OpType::kOperatorReLU) {
}

void ReLUOperator::set_threshold(float threshold) {
    this->threshold_ = threshold;
}

float ReLUOperator::get_threshold() const {
    return threshold_;
}

}
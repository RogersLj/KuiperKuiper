#include <glog/logging.h>
#include "layer/expression_layer.hpp"
#include <stack>
#include "data/tensor.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer {
    
ExpressionLayer::ExpressionLayer(const std::shared_ptr<Operator> &op) : Layer("ExpressionLayer") {
    CHECK(op != nullptr && op->op_type_ == OpType::kOperatorExpression);
    ExpressionOp *expression_op = dynamic_cast<ExpressionOp*>(op.get());

    CHECK(expression_op != nullptr) << "Expression op is empty!";

    this->op_ = std::make_unique<ExpressionOp>(*expression_op);
}

// 对于expression layer，可能设计多个操作，例如add，mul
// 因此对于inputs,不只是一个张量,而是多个顺序排布的张量
//      input 1      |      input 2
// 1.1 1.2 1.3 1.4      2.1 2.2 2.3 2.4
// --------batch size = 4 -------------

void ExpressionLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    
    CHECK(!inputs.empty());

    // 输出张量应初始化大小
    const uint32_t batch_size = outputs.size();
    CHECK(batch_size != 0);

    for (uint32_t i = 0; i < batch_size; ++i) {
        // 初始化输出
        CHECK(outputs.at(i) != nullptr && !outputs.at(i)->empty());
        outputs.at(i)->Fill(0.f);
    }

    CHECK(this->op_ != nullptr && this->op_->op_type_ == OpType::kOperatorExpression);
    // 栈存储的是一个一个batch的输入
    std::stack<std::vector<std::shared_ptr<Tensor<float>>>> oprand_stack;
    // 遍历token列表，遇到数字就入栈，遇到操作符就将两个栈顶元素出栈，并计算
    const std::vector<std::shared_ptr<TokenNode>>& nodes = this->op_->Generate();

    for (const auto& node : nodes) {
        // 顺序遍历所有node

        if (node->num_index >= 0) {
            // 是个数字，找到对应inputs的位置
            // 默认数据是按照@0 @1 @2 @3排列，对应num_index = 0 1 2 3
            uint32_t start_pos = node->num_index * batch_size; // 每个输入的开始位置，batch的第一个元素位置
            std::vector<std::shared_ptr<Tensor<float>>> input_nodes;

            for (uint32_t i = 0; i < batch_size; ++i) {
                CHECK(start_pos + i < inputs.size());
                input_nodes.push_back(inputs.at(start_pos + i));
            }

            oprand_stack.push(input_nodes); 
        } else {
            // 遇到运算符，弹出两个栈顶元素，并计算
            CHECK(oprand_stack.size() >= 2) << "oprand_stack.size() < 2";
            std::vector<std::shared_ptr<Tensor<float>>> input_nodes1 = oprand_stack.top();
            oprand_stack.pop();
            std::vector<std::shared_ptr<Tensor<float>>> input_nodes2 = oprand_stack.top();
            oprand_stack.pop();

            CHECK(input_nodes1.size() == input_nodes2.size());
            
            std::vector<std::shared_ptr<Tensor<float>>> output_nodes(batch_size);
            for (uint32_t i = 0; i < batch_size; ++i) {
                if (node->num_index == -int(TokenType::TokenAdd)) {
                    output_nodes.at(i) = TensorElementAdd(input_nodes1.at(i), input_nodes2.at(i));
                } else if(node->num_index == -int(TokenType::TokenMul)) {
                    output_nodes.at(i) = TensorElementMultiply(input_nodes1.at(i), input_nodes2.at(i));
            } else {
                    LOG(FATAL) << "Unknwon operator";
                    }
                }
            oprand_stack.push(output_nodes);
        }
    }

    CHECK(oprand_stack.size() == 1);

    std::vector<std::shared_ptr<Tensor<float>>> output_nodes = oprand_stack.top();
    oprand_stack.pop();

    for (uint32_t i = 0; i < batch_size; ++i) {
        outputs.at(i) = output_nodes.at(i);
    }

}


}
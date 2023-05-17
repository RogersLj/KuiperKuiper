#include <glog/logging.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer {
    
ReLULayer::ReLULayer(const std::shared_ptr<Operator> &op) : Layer("ReLU") {
    CHECK(op->op_type_ == OpType::kOperatorReLU) << 
        "Operator " << int(op->op_type_)  << " is not " << "ReLU!";
    
    // dynamic_cast关键字用于执行运行时类型检查，并安全地将基类的指针或引用转换为派生类的指针或引用。
    // 将指针 op 强制转换为类型为 ReluOperator * 的指针
    // 在需要将基类的指针或引用转换为派生类的指针或引用时，使用 dynamic_cast。它通过执行检查确保运行时转换是安全的。如果转换不可能，则 dynamic_cast 返回空指针。
    ReLUOperator* relu_op = dynamic_cast<ReLUOperator*>(op.get());

    CHECK(relu_op != nullptr) << "ReLU operator is empty!";

    // 创建了一个ReLUOperator对象的指针
    // 同时使用构造函数进行初始化
    // 函数原型 explicit ReLUOperator(float threshold);
    this->op_ = std::make_unique<ReLUOperator>(relu_op->get_threshold());
}

// 实际的算子计算过程
void ReLULayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    CHECK(this->op_ != nullptr);
    CHECK(this->op_->op_type_ == OpType::kOperatorReLU);
    CHECK(!inputs.empty());

    const uint32_t batch_size = inputs.size();

    for (uint32_t i = 0; i < batch_size; ++i) {
        CHECK(!inputs.at(i)->empty());
        const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
        std::shared_ptr<Tensor<float>> output_data = TensorClone(input_data);

        output_data->data().transform([&](float value) {
            float threshold = this->op_->get_threshold();

            if (value >= threshold) {
                return value;
            } else return 0.f;
        });

        outputs.push_back(output_data);
    }

}

std::shared_ptr<Layer> ReLULayer::CreateInstance(const std::shared_ptr<Operator> &op) {
    CHECK(op->op_type_ == OpType::kOperatorReLU);
    std::shared_ptr<Layer> relu_layer = std::make_shared<ReLULayer>(op);
    return relu_layer;
}

}
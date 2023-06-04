#include <glog/logging.h>
#include "./relu_layer.hpp"
#include "data/tensor_util.hpp"
#include "layer/layer_factory.hpp"
#include "runtime/runtime_operator.hpp"

namespace kuiper_infer {
// 实际的算子计算过程
void ReLULayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) {

    if (inputs.empty()) {
        LOG(ERROR) << "inputs is empty!";
    }

    if (inputs.size() != outputs.size()) {
        LOG(ERROR) << "the input size is not equal to the output size!";
    }

    const uint32_t batch_size = inputs.size();

    for (uint32_t i = 0; i < batch_size; ++i) {
        CHECK(!inputs.at(i)->empty());
        const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
        std::shared_ptr<Tensor<float>> output_data = outputs.at(i);

        output_data->set_data(input_data->data());
        output_data->data().transform([&](float v) { return v > 0.f ? v : 0.f; });
    }
}

void ReLULayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer>& relu_layer) {
    CHECK(op != nullptr) << "Operator is nullptr!";
    relu_layer = std::make_shared<ReLULayer>();
}


// 定已完成之后直接调用LayerRegistererWrapper类初始化实例kReLULayer
// 初始化时会直接在注册表注册
LayerRegisterWrapper kReLULayerCreateInstance("nn.ReLU", ReLULayer::CreateInstance);

}
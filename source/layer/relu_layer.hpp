#ifndef KUIPER_INFER_LAYER_RELU_LAYER_HPP
#define KUIPER_INFER_LAYER_RELU_LAYER_HPP

#include "layer/layer.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {
    
class ReLULayer : public Layer {
    
public:

    // 没有任何参数的算子
    ReLULayer() : Layer("ReLULayer") {};

    // ReLu算子的前向推断
    // override表示覆盖父类虚函数
    void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    // 相同名字，不同属性，有很多relu算子，每个relu有不同的threshold
    static void CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer>& relu_layer);

};
}


#endif
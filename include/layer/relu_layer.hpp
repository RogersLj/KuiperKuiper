#ifndef KUIPER_INFER_LAYER_RELU_LAYER_HPP
#define KUIPER_INFER_LAYER_RELU_LAYER_HPP

#include "layer/layer.hpp"
#include "ops/relu_op.hpp"

namespace kuiper_infer {
    
class ReLULayer : public Layer {
    
public:

    explicit ReLULayer(const std::shared_ptr<Operator> &op);

    // ReLu算子的前向推断
    // override表示覆盖父类虚函数
    void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    // 相同名字，不同属性，有很多relu算子，每个relu有不同的threshold
    static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

private:

    // 用于存放ReLu算子的属性，通过op_获取threshold
    std::unique_ptr<ReLUOperator> op_; 
};
}


#endif
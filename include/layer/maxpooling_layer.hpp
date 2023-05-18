#ifndef KUIPER_INFER_LAYER_MAXPOOLING_LAYER_HPP
#define KUIPER_INFER_LAYER_MAXPOOLING_LAYER_HPP

#include "layer.hpp"
#include "ops/maxpooling_op.hpp"

namespace kuiper_infer {
    
class MaxPoolingLayer : public Layer {
    
public:
    MaxPoolingLayer(const std::shared_ptr<Operator> &op);

    void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    // static 表示可以在不创建实例的情况下直接通过类调用
    static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

private:

    std::unique_ptr<MaxPoolingOp> op_;

};

}


#endif
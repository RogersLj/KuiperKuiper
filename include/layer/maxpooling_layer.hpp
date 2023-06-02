#ifndef KUIPER_INFER_LAYER_MAXPOOLING_LAYER_HPP
#define KUIPER_INFER_LAYER_MAXPOOLING_LAYER_HPP

#include "layer.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {
    
class MaxPoolingLayer : public Layer {
    
public:
    explicit MaxPoolingLayer(Shape kernel_size, Shape stride, Shape padding);

    void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    // static 表示可以在不创建实例的情况下直接通过类调用
    static void CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer>& maxpooling_layer);

private:

    Shape kernel_size_;
    Shape stride_;
    Shape padding_ = {0, 0};

};

}


#endif
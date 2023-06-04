#ifndef KUIPER_INFER_LAYER_ADAPTIVE_AVGPOOLING_HPP
#define KUIPER_INFER_LAYER_ADAPTIVE_AVGPOOLING_HPP

#include "layer/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {

class AdaptiveAvgPoolingLayer : public Layer {
    
public:

    explicit AdaptiveAvgPoolingLayer(uint32_t output_h, uint32_t output_w);

    void Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

    static void CreateInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& adaptive_avgpooling_layer);

private:
    uint32_t output_h_ = 0;
    uint32_t output_w_ = 0;
};

}

#endif
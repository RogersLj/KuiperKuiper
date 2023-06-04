#ifndef KUIPER_INFER_LAYER_CONV_LAYER_HPP
#define KUIPER_INFER_LAYER_CONV_LAYER_HPP

#include "layer/param_layer.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {

class ConvLayer : public ParamLayer {
public:
    explicit ConvLayer(uint32_t in_channels, uint32_t out_channels, Shape kernel_size, Shape stride, Shape padding, uint32_t groups, bool has_bias);
    
    void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static void CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer>& conv_layer);

private:

    bool has_bias_;
    uint32_t groups_ = 1;
    Shape kernel_size_;
    Shape stride_;
    Shape padding_;

};
  
}



#endif
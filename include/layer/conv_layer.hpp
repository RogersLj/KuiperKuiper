#ifndef KUIPER_INFER_LAYER_CONV_LAYER_HPP
#define KUIPER_INFER_LAYER_CONV_LAYER_HPP

#include "layer.hpp"
#include "ops/conv_op.hpp"

namespace kuiper_infer {

class ConvLayer : public Layer {
public:
    ConvLayer(const std::shared_ptr<Operator> &op);
    void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

private:

    std::unique_ptr<ConvOp> op_;


};
  
}



#endif
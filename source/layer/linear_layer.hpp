#ifndef KUIPER_INFER_LAYER_LINEAR_LAYER_HPP
#define KUIPER_INFER_LAYER_LINEAR_LAYER_HPP

#include "layer/layer.hpp"
#include "layer/param_layer.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {
    
class LinearLayer : public ParamLayer {
    
public:

    explicit LinearLayer(uint32_t in_features, uint32_t out_features, bool has_bias);

    void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static void CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer>& linear_layer);

private:

    uint32_t in_features_;
    uint32_t out_features_;
    bool has_bias_;


};


}





#endif
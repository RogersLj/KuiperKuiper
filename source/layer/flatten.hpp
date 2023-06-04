#ifndef KUIPER_INFERENCE_LAYER_FLATTEN_HPP
#define KUIPER_INFERENCE_LAYER_FLATTEN_HPP

#include "layer/layer.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {
    
class FlattenLayer : public Layer {
    
public:

    explicit FlattenLayer(int start_dim, int end_dim);

    void Forward(const std::vector<std::shared_ptr<Tensor<float>>>&inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static void CreateInstance(const std::shared_ptr<RuntimeOperator>& op,  std::shared_ptr<Layer>& flatten_layer);

private:
    int start_dim_ = 0;
    int end_dim_ = 0;

};

}



#endif
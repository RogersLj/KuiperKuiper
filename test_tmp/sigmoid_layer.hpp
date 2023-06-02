#ifndef KUIPER_INFER_LAYER_SIGMOID_LAYER_HPP
#define KUIPER_INFER_LAYER_SIGMOID_LAYER_HPP

#include "layer/layer.hpp"
#include "ops/sigmoid_op.hpp"

namespace kuiper_infer {
    
class SigmoidLayer : public Layer {
public:
    explicit SigmoidLayer(const std::shared_ptr<Operator> &op);

    void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

private:
    std::unique_ptr<SigmoidOperator> op_;
    // 每次都是指向一个新的op

};

}


#endif
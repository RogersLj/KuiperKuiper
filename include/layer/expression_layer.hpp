#ifndef KUIPER_INFER_LAYER_EXPRESSION_LAYER_HPP
#define KUIPER_INFER_LAYER_EXPRESSION_LAYER_HPP

#include "layer.hpp"
#include "ops/expression_op.hpp"


namespace kuiper_infer {
    
class ExpressionLayer : public Layer {
    
public:

    explicit ExpressionLayer(const std::shared_ptr<Operator> &op);

    void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

private:

    std::unique_ptr<ExpressionOp> op_;

};


};



#endif
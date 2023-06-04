#ifndef KUIPER_INFER_LAYER_EXPRESSION_LAYER_HPP
#define KUIPER_INFER_LAYER_EXPRESSION_LAYER_HPP

#include "layer/layer.hpp"
#include "runtime/runtime_ir.hpp"
#include "parser/parse_expression.hpp"


namespace kuiper_infer {
    
class ExpressionLayer : public Layer {
    
public:

    explicit ExpressionLayer(const std::string& expression);

    void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

    static void CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer>& expression_layer);

private:
    // parser里的Generate()方法
    std::shared_ptr<ExpressionParser> parser_; // 用于构建计算图的解析器

};


};



#endif
#ifndef KUIPER_INFER_LAYER_HPP
#define KUIPER_INFER_LAYER_HPP

#include <string>
#include "data/tensor.hpp"

namespace kuiper_infer {

// 所有算子的实际计算由Layer类定义

class Layer {
public:
    explicit Layer(const std::string &layer_name);

    // 输入是一个 batch 的 Tensor
    virtual void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs);

    virtual ~Layer() = default;


private:
    std::string layer_name_; // layer 的名字
};

}


#endif
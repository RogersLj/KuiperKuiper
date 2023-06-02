#ifndef KUIPER_INFER_LAYER_HPP
#define KUIPER_INFER_LAYER_HPP

#include <string>
#include "data/tensor.hpp"

namespace kuiper_infer {

// 所有算子的实际计算由Layer类定义

typedef std::pair<uint32_t, uint32_t> Shape;

class Layer {
public:
    explicit Layer(const std::string &layer_name) : layer_name_(std::move(layer_name)) {};

    // 输入是一个 batch 的 Tensor
    virtual void Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs);

    virtual ~Layer() = default;

    virtual const std::vector<std::shared_ptr<Tensor<float>>> &weights() const;

    virtual const std::vector<std::shared_ptr<Tensor<float>>> &bias() const;

    virtual void set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights);

    virtual void set_weights(const std::vector<float>& weights);

    virtual void set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias);

    virtual void set_bias(const std::vector<float>& bias);

    virtual const std::string& layer_name() const { return layer_name_; }

protected:
    std::string layer_name_; // layer 的名字
};

}


#endif
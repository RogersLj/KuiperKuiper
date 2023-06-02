#ifndef KUIPER_INFER_OPS_CONV_HPP
#define KUIPER_INFER_OPS_CONV_HPP

#include "op.hpp"
#include <cstdint>
#include <vector>
#include "data/tensor.hpp"
#include <utility>

namespace kuiper_infer {

typedef std::pair<uint32_t, uint32_t> Shape;
    
class ConvOp : public Operator {

public:

    explicit ConvOp(Shape stride, Shape padding, bool has_bias, uint32_t groups) : Operator(OpType::kOperatorConv), stride_(stride), padding_(padding), has_bias_(has_bias), groups_(groups) {};

    void set_stride(Shape stride);

    void set_padding(Shape padding);

    void set_has_bias(bool has_bias);

    void set_groups(uint32_t groups);

    Shape get_stride() const;

    Shape get_padding() const;

    bool get_has_bias() const;

    uint32_t get_groups() const;

    void set_weights(std::vector<sftensor> &weights);

    void set_bias(std::vector<sftensor> &bias);

    const std::vector<sftensor>& get_weights() const;

    const std::vector<sftensor>& get_bias() const;

private:

    bool has_bias_ = false;
    uint32_t groups_ = 1;
    Shape stride_;
    Shape padding_;
    std::vector<std::shared_ptr<Tensor<float>>> weights_;
    std::vector<std::shared_ptr<Tensor<float>>> bias_;


};
}


#endif
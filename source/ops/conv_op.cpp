#include "ops/conv_op.hpp"


namespace kuiper_infer {
    
bool ConvOp::get_has_bias() const {
    return this->has_bias_;
}

uint32_t ConvOp::get_groups() const {
    return this->groups_;
}

Shape ConvOp::get_stride() const {
    return this->stride_;
}

Shape ConvOp::get_padding() const {
    return this->padding_;
}

void ConvOp::set_stride(Shape stride) {
    this->stride_ = stride;
}

void ConvOp::set_padding(Shape padding) {
    this->padding_ = padding;
}

void ConvOp::set_has_bias(bool has_bias) {
    this->has_bias_ = has_bias;
}

void ConvOp::set_groups(uint32_t groups) {
    this->groups_ = groups;
}

void ConvOp::set_weights(std::vector<sftensor> &weights) {
    this->weights_ = weights;
}

void ConvOp::set_bias(std::vector<sftensor> &bias) {
    this->bias_ = bias;
}

const std::vector<sftensor>& ConvOp::get_weights() const {
    return this->weights_;
}

const std::vector<sftensor>& ConvOp::get_bias() const {
    return this->bias_;
}

}
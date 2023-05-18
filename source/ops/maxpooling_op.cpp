#include "ops/maxpooling.hpp"

namespace kuiper_infer {
    
MaxPoolingOp::MaxPoolingOp(Shape kernel_size, Shape stride, Shape padding) : Operator(OpType::kOperatorMaxPooling), kernel_size_(kernel_size), stride_(stride), padding_(padding) {}


void MaxPoolingOp::set_kernel_size(Shape kernel_size) {
    this->kernel_size_ = kernel_size;
}

void MaxPoolingOp::set_stride(Shape stride) {
    this->stride_ = stride;
}

void MaxPoolingOp::set_padding(Shape padding) {
    this->padding_ = padding;
}

Shape MaxPoolingOp::get_kernel_size() const {
    return kernel_size_;
}

Shape MaxPoolingOp::get_stride() const {
    return stride_;
}

Shape MaxPoolingOp::get_padding() const {
    return padding_;
}

}
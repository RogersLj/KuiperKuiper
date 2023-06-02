#ifndef KUIPER_INFER_OPS_MAXPOOLING_HPP
#define KUIPER_INFER_OPS_MAXPOOLING_HPP

#include "op.hpp"
#include <cstdint>
#include <utility>

namespace kuiper_infer {
    
typedef std::pair<uint32_t, uint32_t> Shape;

class MaxPoolingOp : public Operator {

public:

    MaxPoolingOp(Shape kernel_size, Shape stride, Shape padding);

    void set_kernel_size(Shape kernel_size);

    void set_stride(Shape stride);

    void set_padding(Shape padding);

    Shape get_kernel_size() const;

    Shape get_stride() const;

    Shape get_padding() const;


private:

    Shape kernel_size_;
    Shape stride_;
    Shape padding_;
};

}

#endif



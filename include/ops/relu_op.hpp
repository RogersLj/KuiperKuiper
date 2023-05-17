#ifndef KUIPER_INFER_OPS_RELU_OP_H
#define KUIPER_INFER_OPS_RELU_OP_H

#include "./op.hpp"

namespace kuiper_infer {

class ReLUOperator : public Operator {
// 对于ReLu算子，在存储属性的时候，只需要存储阈值
// 以及设置阈值和获取阈值的函数
// 实际的计算在 Layer 中定义
public:
    explicit ReLUOperator(float threshold);

    void set_threshold(float threshold);

    float get_threshold() const;

private:
    float threshold_ = 0.f;

};

}

#endif
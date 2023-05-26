#ifndef KUIPER_INFER_OPS_OP_HPP
#define KUIPER_INFER_OPS_OP_HPP

namespace kuiper_infer {

// 和传统C语言的enum不同，使用时需使用OpType::
enum class OpType {
    kOperatorUnknown = -1,
    kOperatorReLU = 0,
    kOperatorSigmoid = 1,
    kOperatorMaxPooling = 2,
    kOperatorExpression = 3,
};

// 所有算子的父类
class Operator {

public:
    OpType op_type_ = OpType::kOperatorUnknown; // 算子类型，默认是unknown

    explicit Operator() = default;
    
    explicit Operator(OpType op_type);

    virtual ~Operator() = default;

};





}
#endif
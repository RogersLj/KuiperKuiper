#ifndef KUIPER_INFER_RUNTIME_OPERAND_HPP
#define KUIPER_INFER_RUNTIME_OPERAND_HPP

#include <vector>
#include <string>
#include <memory>
#include "status_code.hpp"
#include "runtime_datatype.hpp"
#include "data/tensor.hpp"


namespace kuiper_infer {
// 操作数，张量

struct RuntimeOperand {
    std::string name; // 操作数名
    std::vector<int32_t> shape; // 操作数的形状
    RuntimeDataType type = RuntimeDataType::kTypeUnknown; // 操作数的类型 - float
    std::vector<std::shared_ptr<Tensor<float>>> datas; // 操作数的张量 - 运行时存储实际数据 - 一个batch的数据
};

}


#endif

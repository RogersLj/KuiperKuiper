#ifndef KUIPER_INFER_RUNTIME_OPERATOR_HPP
#define KUIPER_INFER_RUNTIME_OPERATOR_HPP


#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <string>
#include "factory/layer_factory.hpp"
#include "runtime_operand.hpp"
#include "runtime_attr.hpp"
#include "runtime_param.hpp"

namespace kuiper_infer {
    
// 运算符，算子
class Layer;

struct RuntimeOperator {
    
    ~RuntimeOperator() {
        for (const auto& param : this->params) {
            delete param.second;
        }
    };

    std::string name; // 算子名 - conv1
    std::string type; // 算子类型
    std::shared_ptr<Layer> layer; // 算子计算的层 - 实际计算的算子

    // 输入节点可能有多个，输入操作数有多个
    std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands; // 输入操作数,名字是前一个算子节点的名字
    std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq; // 顺序排列的算子输入操作数

    // 输出节点可能有多个，但是输出值只有一个
    std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators; // 输出节点名字和对应节点
    std::vector<std::string> output_names; // 输出节点名字
    std::shared_ptr<RuntimeOperand> output_operands; // 输出操作数

    std::map<std::string, RuntimeParameter*> params; //算子的参数信息，例如conv的参数表有kernel_size {3,3}, stride, padding
    std::map<std::string, std::shared_ptr<RuntimeAttribute>> attrs;
    // 算子属性 - 权重信息，训练之后确定
};

}

#endif
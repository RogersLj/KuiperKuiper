#ifndef KUIPER_INFER_RUNTIME_OPERATOR_HPP
#define KUIPER_INFER_RUNTIME_OPERATOR_HPP


#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <string>
#include "layer/layer_factory.hpp"
#include "runtime_operand.hpp"
#include "runtime_attr.hpp"
#include "runtime_param.hpp"

namespace kuiper_infer {
    
// 运算符，算子
class Layer;

struct RuntimeOperator {

    int32_t meet_time = 0; // 被前面算子填充输入的次数，当该次数等于输入节点时，即可加入执行队列
    
    ~RuntimeOperator() {
        for (const auto& param : this->params) {
            delete param.second;
        }
    };

    std::string name; // 算子名 - conv1
    std::string type; // 算子类型
    std::shared_ptr<Layer> layer; // 算子计算的层 - 实际计算的算子

    // 输入节点可能有多个，输入操作数有多个
    std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands; // 输入操作数,名字是前一个算子节点的名字(来自的节点，输入操作数)，用于检查该节点是否可以加入执行队列里
    std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq; // 顺序排列的算子输入操作数，存的指针，不太占内存

    // 输出节点可能有多个，但是输出值只有一个
    std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators; // 输出节点名字和对应节点
    std::vector<std::string> output_names; // 输出节点名字
    std::shared_ptr<RuntimeOperand> output_operand; // 输出操作数 - 一个

    std::map<std::string, RuntimeParameter*> params; //算子的参数信息，例如conv的参数表有kernel_size {3,3}, stride, padding
    std::map<std::string, std::shared_ptr<RuntimeAttribute>> attrs;
    // 算子属性 - 权重信息，训练之后确定
};

}

#endif
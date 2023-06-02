#include <vector>
#include <string>
#include <glog/logging.h>
#include <memory>
#include <map>
#include <queue>

#include "ir.h"
#include "factory/layer_factory.hpp"
#include "runtime/runtime_operand.hpp"
#include "runtime_operator.hpp"



namespace kuiper_infer {
    

// 因为在没有做任何并行优化的情况下，一个batch的数据是一个一个处理的
// 假设batch里每个数据大小都一样，因此在前向推理的过程中，中间输出结果的大小也是固定的，只是值不一样
// 为了节省时间,在第一次构建计算图的时候分配好存储中间数据的内存空间
class RuntimeGraphShape {

public:

    static void InitOpeartorInputTensor(const std:: vector<std::shared_ptr<RuntimeOperator>>& operators);


    static void InitOperatorOutputTensor(const std::vector<pnnx::Operator*> &pnnx_operators, const std::vector<std::shared_ptr<RuntimeOperator>>& operators);

};

// 定义的计算图结构
class RuntimeGraph {

public:

/*
初始化计算图
@return 初始化成功返回true
*/
// 主要是初始化operator,operands,attrs,params
    bool Init();

/*
构建计算图
@param input_name 输入名字
@param output_name 输出名字
*/
    void Build(const std::string& input_name, const std::string& output_name); // 根据输入节点和输出节点构建完整计算图


/*
设置权重信息
@param param_path 计算图的结构文件路径
@param bin_path 计算图的权重文件路径
*/
    RuntimeGraph(std::string param_path, std::string bin_path);

// 权重文件和图结构地址
    void set_param(const std::string& param_path);

    void set_bin(const std::string& bin_path);

    const std::string& param_path() const;

    const std::string& bin_path() const;

// 返回算子列表
    const std::vector<std::shared_ptr<RuntimeOperator>> operators() const;


// 前向推理 - 最终的推理

    std::vector<std::shared_ptr<Tensor<float>>> Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, bool debug = false);



//  所有的static函数只能通过类public函数访问
private:
    // 转化pnnx op时,同时为每个RuntimeOperator初始化layer
    // attrs需要装载到layer里
    static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<RuntimeOperator>& op);

    // 检查是否当前算子输入准备完成，可以加入执行队列
    static bool CheckOperatorReady(const std::shared_ptr<RuntimeOperator>& op);

    // 上一个节点的输出操作数搬运到下一个节点的输入操作数，一个输出可能会送到多个算子
    static void SetOpInputData(const std::vector<std::shared_ptr<Tensor<float>>>& src, std::vector<std::vector<std::shared_ptr<Tensor<float>>>>& dst);

    // 将输出传给下一个节点作为输入操作数
    static void PassOutputToNext(const std::shared_ptr<RuntimeOperator>& cur_op, std::deque<std::shared_ptr<RuntimeOperator>>& operators_queue, std::vector<std::shared_ptr<Tensor<float>>> layer_output_data);

    // 根据节点的输入构建算子
    static void InitOperatorInputs(const std::vector<pnnx::Operand*>& inputs,
                                   const std::shared_ptr<RuntimeOperator>& runtime_operator);

    // 根据节点的输出构建算子
    static void InitOperatorOutputs(const std::vector<pnnx::Operand*>& outputs,
                                    const std::shared_ptr<RuntimeOperator>& runtime_operator);

    // 构建算子参数
    static void InitGraphParams(const std::map<std::string, pnnx::Parameter>& params,
                                const std::shared_ptr<RuntimeOperator>& runtime_operator);

    // 构建算子属性
    static void InitGraphAttrs(const std::map<std::string, pnnx::Attribute>& attrs,
                               const std::shared_ptr<RuntimeOperator>& runtime_operator);


private:

    enum class GraphState {
        kToInit = -2,
        kToBuild = -1,
        kComplete = 0,
    };

    GraphState graph_state_ = GraphState::kToInit;
    std::string input_name_;
    std::string output_name_;
    std::string param_path_;
    std::string bin_path_;

    std::map<std::string, std::shared_ptr<RuntimeOperator>> input_operators_map_; // 输入节点 - 生产者
    std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators_map_; // 输出节点 - 消费者
    std::vector<std::shared_ptr<RuntimeOperator>> operators_; // 算子集合
    std::unique_ptr<pnnx::Graph> graph_; // PNNX 计算图

};

}
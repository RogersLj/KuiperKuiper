#include "runtime/runtime_ir.hpp"
#include <memory>
#include <iostream>
#include <iomanip>
#include <queue>
#include <utility>
#include "factory/layer_factory.hpp"

namespace kuiper_infer {

void RuntimeGraphShape::InitOpeartorInputTensor(const std:: vector<std::shared_ptr<RuntimeOperator>>& operators) {
    if (operators.empty()) {
        LOG(ERROR) << "operators for init is empty";
        return;
    }

    for (const auto& op : operators) {
        // 遍历所有的operators,并初始化input_oprands
        if (op->input_operands.empty()) {
            continue; // 输出节点就没有输入操作数
        } else {
            const std::map<std::string, std::shared_ptr<RuntimeOperand>>& input_operands_map = op->input_operands;

            for (const auto& input_operand_iter : input_operands_map) {
                const auto& input_operand_name = input_operand_iter.first;
                const auto& input_operand = input_operand_iter.second;
                const auto& type = input_operand->type;
                CHECK(type == RuntimeDataType::kTypeFloat32) << "The inference graph only support float32 data type now";
                const auto& input_operand_shape = input_operand->shape;
                auto& input_datas = input_operand->datas;

                CHECK(!input_operand_shape.empty());
                const int32_t batch_size = input_operand_shape.at(0);

                CHECK(batch_size >= 0) << "Dynamic batch_size size is not supported!";

                CHECK(input_operand_shape.size() == 2 ||
                        input_operand_shape.size() == 4 ||
                        input_operand_shape.size() == 3)
                        << "Unsupported tensor shape sizes: " << input_operand_shape.size();

                // 第一次运行，没有初始化，初始化一个batch_size的数据空间
                // 第二次运行，如果数据已经填充，检查填充的数据大小和预期是否一致
                if (!input_datas.empty()) {
                    CHECK(input_datas.size() == batch_size) << "The size of input data is not equal to the batch_size size!";

                    for (int32_t i = 0; i < batch_size; ++i) {
                        const std::vector<uint32_t>& input_data_shape = input_datas.at(i)->shape(); // CHW
                        CHECK(input_data_shape.size() == 3) << "The original shape of input data is not 3!";

                        // input_operand_shape : (batch_size, elemsize) 一维的
                        // input_operand_shape : (batch_size, rows, cols) 二维的
                        // input_operand_shape : (batch_size, channels, rows, cols) 三维的
                        // input_data_shape : (channels, rows, cols) - 看起来始终是三维的
                        // input_data_shape : (1, rows, cols) - 二维时
                        // input_data_shape : (1, 1, cols) - 一维时
                        if (input_operand_shape.size() == 4) {
                            CHECK(input_data_shape.at(0) == input_operand_shape.at(1) && input_data_shape.at(1) == input_operand_shape.at(2) && input_data_shape.at(2) == input_operand_shape.at(3));
                        } else if (input_operand_shape.size() == 2) {
                            // 一维数据
                            CHECK(input_data_shape.at(2) == input_operand_shape.at(1) && input_data_shape.at(0) == input_data_shape.at(1) == 1);
                        } else {
                            // 二维数据
                            CHECK(input_data_shape.at(0) == 1 && input_data_shape.at(1) == input_operand_shape.at(1) && input_data_shape.at(2) == input_operand_shape.at(2));
                        }   
                    }
                } else {
                    // 第一次执行，分配空间
                    input_datas.resize(batch_size);
                    for (int32_t i = 0; i < batch_size; ++i) {
                        if (input_operand_shape.size() == 4) {
                            // 三维数据
                            input_datas.at(i) = std::make_shared<Tensor<float>>(input_operand_shape.at(1), input_operand_shape.at(2), input_operand_shape.at(3));
                        } else if (input_operand_shape.size() == 2) {
                            // 一维数据
                            input_datas.at(i) = std::make_shared<Tensor<float>>(1, 1, input_operand_shape.at(1));
                        } else {
                            // 一维数据
                            input_datas.at(i) = std::make_shared<Tensor<float>>(1, input_operand_shape.at(1), input_operand_shape.at(2));
                        }
                    }
                }
            }
        }
    }
}

// 因为初始化RuntimeOperator的时候对于operator的输出，只添加了节点的输出名字和对应节点，没有初始化output_operand
// 因此这里需要用原本的pnxx图里的operator信息来为输出分配空间，并初始化节点的output_operand
void RuntimeGraphShape::InitOperatorOutputTensor(const std::vector<pnnx::Operator*> &pnnx_operators, const std::vector<std::shared_ptr<RuntimeOperator>>& operators) {

    CHECK(!pnnx_operators.empty() && !operators.empty());
    CHECK(pnnx_operators.size() == operators.size());

    for (int32_t i = 0; i < pnnx_operators.size(); ++i) {
        // 当前operator的输出操作数，每一个操作数是一个batch_size的大小
        const std::vector<pnnx::Operand*> operands = pnnx_operators.at(i)->outputs;

        CHECK(operands.size() <= 1) << "Only support one output now";
        if (operands.empty()) {
            continue;
        }

        CHECK(operands.size() == 1) << "Only support one output now";

        pnnx::Operand* operand = operands.front();
        const auto& runtime_operator = operators.at(i);
        CHECK(operand != nullptr) << "operand is nullptr";
        const std::vector<int32_t> shape = operand->shape;
        const auto& output_operand = runtime_operator->output_operand;

        const int32_t batch_size = shape.at(0); // batch_size大小
        CHECK(batch_size >= 0) << "Dynamic shape is not supported now";
        CHECK(shape.size() == 2 || shape.size() == 3 || shape.size() == 4) << "Dynamic shape is not supported now";

        if (!output_operand) {
            // 还没有初始化节点的输出
            std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();

            runtime_operand->shape = shape;
            runtime_operand->name = operand->name + "_output";
            runtime_operand->type = RuntimeDataType::kTypeFloat32;

            for (int32_t j = 0; j < batch_size; j ++) {
                if (shape.size() == 4) {
                    runtime_operand->datas.push_back(std::make_shared<Tensor<float>>(shape.at(1), shape.at(2), shape.at(3)));
                } else if (shape.size() == 2) {
                    runtime_operand->datas.push_back(std::make_shared<Tensor<float>>(1, 1, shape.at(1)));
                } else {
                    runtime_operand->datas.push_back(std::make_shared<Tensor<float>>(1, shape.at(1), shape.at(2)));
                }
            }

            runtime_operator->output_operand = runtime_operand;
        } else {
            // 检查大小是否一致
            CHECK(output_operand->type == RuntimeDataType::kTypeFloat32);
            CHECK(output_operand->shape == shape);

            for (int32_t j = 0; j < batch_size; j ++) {
                const std::vector<uint32_t>& operand_shape = output_operand->datas.at(j)->shape();
                if (shape.size() == 4) {
                    if (operand_shape.at(0) != shape.at(1) || operand_shape.at(1) != shape.at(2) || operand_shape.at(2) != shape.at(3)) {
                        DLOG(WARNING) << "The shape of calculated output tensor do not match with output operand";
                    }
                    const auto& new_shape = std::vector<uint32_t> {(uint32_t)shape.at(1), (uint32_t)shape.at(2), (uint32_t)shape.at(3)};
                    // shape是pnnx训练得到的真实值
                    output_operand->datas.at(j)->Reshape(new_shape);
                } else if (shape.size() == 2) {
                    if (operand_shape.at(0) != 1 || operand_shape.at(1) != 1 || operand_shape.at(2) != shape.at(1)) {
                        DLOG(WARNING) << "The shape of calculated output tensor do not match with output operand";
                    }
                    const auto& new_shape = std::vector<uint32_t> {(uint32_t)1, (uint32_t)1, (uint32_t)shape.at(1)};
                    output_operand->datas.at(j)->Reshape(new_shape);
                } else {
                    if (operand_shape.at(0) != 1 || operand_shape.at(1) != shape.at(1) || operand_shape.at(2) != shape.at(2)) {
                        DLOG(WARNING) << "The shape of calculated output tensor do not match with output operand";
                    }
                    const auto& new_shape = std::vector<uint32_t> {(uint32_t)1, (uint32_t)shape.at(1), (uint32_t)shape.at(2)};
                    output_operand->datas.at(j)->Reshape(new_shape);
                }
            }
        }

    }

}

    
RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path) : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {
    std::cout << "param_path: " << this->param_path_ << std::endl;
    std::cout << "bin_path: " << this->bin_path_ << std::endl;
}

void RuntimeGraph::set_param(const std::string& param_path) {
    this->param_path_ = param_path;
}

void RuntimeGraph::set_bin(const std::string& bin_path) {
    this->bin_path_ = bin_path;
}

const std::string& RuntimeGraph::param_path() const {
    return this->param_path_;
}

const std::string& RuntimeGraph::bin_path() const {
    return this->bin_path_;
}


// pnnx::Graph里有操作数表和算子表
// 但实际上每个Operator和每个Operand都相互指向
// 因此只需要根据Operator列表或Operand列表就可以构建计算图
// 这里根据Operator列表构建计算图
// 图初始化函数，将pnnx::operands转换为RuntimeOperand，将pnnx::operators转换为RuntimeOperator
bool RuntimeGraph::Init() {
    
    if (this->param_path_.empty() || this->bin_path_.empty()) {
        LOG(ERROR) << "param_path or bin_path is empty";
        return false;
    }

    this->graph_ = std::make_unique<pnnx::Graph>();
    int load_status = this->graph_->load(this->param_path_, this->bin_path_);
    // 加载pnnx计算图
    std::cout << "load pnnx graph, param_path: " << this->param_path_ << ", bin_path: " << this->bin_path_ << std::endl;
    std::cout << "load_status: " << load_status << std::endl;
    if (load_status != 0) {
        LOG(ERROR) << "load pnnx graph failed, check param_path or bin_path";
        return false;
    }

    std::vector<pnnx::Operator*> operators = this->graph_->ops;
    // 获取计算图的算子列表

    if (operators.empty()) {
        LOG(ERROR) << "can't read graph operators";
        return false;
    }

    this->operators_.clear();
    // 通过pnxx::Operator构建RuntimeOperator
    // pnnx::Operator 的成员变量
    // std::vector<Operand*> inputs;
    // std::vector<Operand*> outputs;
    // std::string type;
    // std::string name;
    // std::vector<std::string> inputnames;
    // std::map<std::string, Parameter> params;
    // std::map<std::string, Attribute> attrs;
    for (const pnnx::Operator* op : operators) {
        if (!op) {
            LOG(ERROR) << "op is null";
            continue;
        } else {
            // 对每一个pnnx::Operator,将其转换为RuntimeOperator
            std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();

            runtime_operator->name = op->name;
            runtime_operator->type = op->type;

            const std::vector<pnnx::Operand*>& inputs = op->inputs;
            if (!inputs.empty()) {
                InitOperatorInputs(inputs, runtime_operator);
            }

            const std::vector<pnnx::Operand*>& outputs = op->outputs;
            if (!outputs.empty()) {
                InitOperatorOutputs(outputs, runtime_operator);
            }


            const std::map<std::string, pnnx::Parameter>& params = op->params;
            if (!params.empty()) {
                InitGraphParams(params, runtime_operator);
            }

            const std::map<std::string, pnnx::Attribute>& attrs = op->attrs;
            if (!attrs.empty()) {
                InitGraphAttrs(attrs, runtime_operator);
            }

            // 初始化好该算子
            this->operators_.push_back(runtime_operator);

        }
    }

    
    for (const auto& cur_op : this->operators_) {
        // 输出节点的名字在InitOperatorOutputs函数里得到
        // 为什么不在得到输出节点的名字的时候就加入{next_op->name, next_op}，因此那个时候，输出节点还有被初始化完成RuntimeOperator，还有param和attr
        const std::vector<std::string>& output_names = cur_op->output_names;

        for (const auto& next_op : this->operators_) {
            if (next_op == cur_op) {
                continue;
            }
            if (std::find(output_names.begin(), output_names.end(), next_op->name) != output_names.end()) {
                // 将输出算子的名字和算子关联
                cur_op->output_operators.insert({next_op->name, next_op});
            }
        }
    }

    graph_state_ = GraphState::kToBuild;

    return true;
}


// 封装后没有生产者和消费者的概念
// 每个算子的输入输出，是上一个或下一个​算子
void RuntimeGraph::InitOperatorInputs(const std::vector<pnnx::Operand*>& inputs, const std::shared_ptr<RuntimeOperator>& runtime_operator) {
    for (const pnnx::Operand* input : inputs) {
        if (!input) {
            continue;
        }

        const pnnx::Operator* producer = input->producer;
        std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
        runtime_operand->name = input->name;
        runtime_operand->shape = input->shape;

        switch (input->type) {
            case 1 : {
                runtime_operand->type = RuntimeDataType::kTypeFloat32;
                break;
            }
            case 0: {
                runtime_operand->type = RuntimeDataType::kTypeUnknown;
                break;
            }
            default: {
                LOG(ERROR) << "unknown input type" << input->type;
            }
        }

        runtime_operator->input_operands.insert({producer->name, runtime_operand});
        runtime_operator->input_operands_seq.push_back(runtime_operand);
        // 注意这里初始化operand时没有初始化operand->data - 因为具体数据是在inference阶段才填充
    }
}

void RuntimeGraph::InitOperatorOutputs(const std::vector<pnnx::Operand*>& outputs, const std::shared_ptr<RuntimeOperator>& runtime_operator) {
    for (const pnnx::Operand* output : outputs) {
        if (!output) {
            continue;
        }
        // 遍历所有生产者就可以得到所有operands，因此不需要将消费者转为RuntimeOperand
        const auto& consumers = output->consumers;
        // 可能会有多个消费者
        for (const auto &c : consumers) {
            runtime_operator->output_names.push_back(c->name);
        }
    }
}

void RuntimeGraph::InitGraphParams(const std::map<std::string, pnnx::Parameter>& params, const std::shared_ptr<RuntimeOperator>& runtime_operator) {
    for (const auto& param : params) {
        const std::string& name = param.first;
        const pnnx::Parameter& parameter = param.second;
        const int type = parameter.type;

// 对应pnnx的参数定义
// 0 - kParameterUnknown: indicates an unknown data type
// 1 - kParameterBool: represents a boolean data type
// 2 - kParameterInt: represents an integer data type
// 3 - kParameterFloat: represents a float data type
// 4 - kParameterString: represents a string data type
// 5 - kParameterIntArray: represents an array of integers
// 6 - kParameterFloatArray: represents an array of floats
// 7 - kParameterStringArray: represents an array of strings.
// 0=null 1=b 2=i 3=f 4=s 5=ai 6=af 7=as 8=others
        switch (type) {
            case int(RuntimeParameterType::kParameterUnknown) : {
                RuntimeParameter* runtime_parameter = new RuntimeParameter;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }

            case int(RuntimeParameterType::kParameterBool) : {
                RuntimeParameterBool* runtime_parameter = new RuntimeParameterBool;
                runtime_parameter->value = parameter.b;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }
            case int(RuntimeParameterType::kParameterInt) : {
                RuntimeParameterInt* runtime_parameter = new RuntimeParameterInt;
                runtime_parameter->value = parameter.i;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }

            case int(RuntimeParameterType::kParameterFloat) : {
                RuntimeParameterFloat* runtime_parameter = new RuntimeParameterFloat;
                runtime_parameter->value = parameter.f;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }

            case int(RuntimeParameterType::kParameterString) : {
                RuntimeParameterString* runtime_parameter = new RuntimeParameterString;
                runtime_parameter->value = parameter.s;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }

            case int(RuntimeParameterType::kParameterIntArray) : {
                RuntimeParameterIntArray* runtime_parameter = new RuntimeParameterIntArray;
                runtime_parameter->value = parameter.ai;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }

            case int(RuntimeParameterType::kParameterFloatArray) : {
                RuntimeParameterFloatArray* runtime_parameter = new RuntimeParameterFloatArray;
                runtime_parameter->value = parameter.af;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }

            case int(RuntimeParameterType::kParameterStringArray) : {
                RuntimeParameterStringArray* runtime_parameter = new RuntimeParameterStringArray;
                runtime_parameter->value = parameter.as;
                runtime_operator->params.insert({name, runtime_parameter});
                break;
            }

            default: {
                LOG(ERROR) << "unknown parameter type" << type;
            }
        }

    }
}

void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute>& attrs, const std::shared_ptr<RuntimeOperator>& runtime_operator) {
    for (const auto& attr : attrs) {
        const std::string& name = attr.first;
        const pnnx::Attribute& attribute = attr.second;

    // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool
    // int type;
    // std::vector<int> shape;
    // std::vector<char> data;

        switch (attribute.type) {
            case 1 : {
                std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>();

                runtime_attribute->type = RuntimeDataType::kTypeFloat32;
                runtime_attribute->shape = attribute.shape;
                runtime_attribute->weight_data = attribute.data;

                runtime_operator->attrs.insert({name, runtime_attribute});
                break;
            }
            default: {
                LOG(FATAL) << "unknown attribute type" << attribute.type;
            }

    }
    }
}


const std::vector<std::shared_ptr<RuntimeOperator>> RuntimeGraph::operators() const {
    // CHECK(graph_state_ == GraphState::kComplete);
    return this->operators_;
}

//void Build(const std::string& input_name, const std::string& output_name);
void RuntimeGraph::Build(const std::string& input_name, const std::string& output_name) {
    if (graph_state_ == GraphState::kToInit) {
        bool init_graph = Init();
        LOG_IF(FATAL, !init_graph) << "Init graph failed!";
    }

    // 初始化之后,得到了所有的operators和operands
    // 以及算子之间的关系

    CHECK(graph_state_ >= GraphState::kToBuild)
        << "Graph status error, current state is " << int(graph_state_);

    LOG_IF(FATAL, this->operators_.empty())
          << "Graph operators is empty, init may failed";

    this->input_operators_map_.clear();
    this->output_operators_map_.clear();

    for (const auto& op : this->operators_) {
        if (op->type == "pnnx.Input") {
            // 所有的输入节点
            this->input_operators_map_.insert({op->name, op});
            } else if (op->type == "pnnx.Output") {
                this->output_operators_map_.insert({op->name, op});
    } else {
      // 以后的课中加layer的
      }
    }

    RuntimeGraphShape::InitOpeartorInputTensor(this->operators_);
    RuntimeGraphShape::InitOperatorOutputTensor(this->graph_->ops, this->operators_);


    this->input_name_ = input_name;
    this->output_name_ = output_name;
    this->graph_state_ = GraphState::kComplete;

}


std::vector<std::shared_ptr<Tensor<float>>> RuntimeGraph::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, bool debug) {
    if (this->graph_state_ < GraphState::kComplete) {
        LOG(FATAL) << "Graph need to be builded first";
    }

    CHECK(this->graph_state_ == GraphState::kComplete) << "Graph status error, current state is " << int(this->graph_state_);

    std::shared_ptr<RuntimeOperator> input_op; // 计算图的入口
    if (input_operators_map_.find(input_name_) == input_operators_map_.end()) {
        LOG(FATAL) << "input operator not found";
    } else {
        input_op = input_operators_map_.at(input_name_);
    }

    std::shared_ptr<RuntimeOperator> output_op;
    if (output_operators_map_.find(output_name_) == output_operators_map_.end()) {
        LOG(FATAL) << "output operator not found";
    } else {
        output_op = output_operators_map_.at(output_name_);
    }

    std::deque<std::shared_ptr<RuntimeOperator>> operators_queue;
    operators_queue.push_back(input_op);
    // 暂时假设计算图的输入只有一个
    // 且输出也只有一个

    if (debug) {
        LOG(INFO) << "Batch size: " << inputs.size();
        for (int i = 0; i < inputs.size(); ++i) {
            LOG(INFO) << "Input Rows: " << inputs.at(i)->rows()
                    << " Cols: " << inputs.at(i)->cols()
                    << " Channels: " << inputs.at(i)->channels();
                    }
        LOG(INFO) << "Inference starting...";
        LOG(INFO) << "--------------------------------------------------" << "\n";
    }

    while (operators_queue.size()) {
        std::shared_ptr<RuntimeOperator> cur_op = operators_queue.front();
        operators_queue.pop_front();

        if (!cur_op || cur_op == output_op) {
            if (debug) {
                LOG(INFO) << "Inference end";
            }
            break;
        }

        if (cur_op == input_op) {
            PassOutputToNext(cur_op, operators_queue, inputs); // 装上输入
        } else {
            std::string cur_op_name = cur_op->name;
            if (!CheckOperatorReady(cur_op)) {
                if (operators_queue.empty()) {
                    LOG(FATAL) << "Operator " << cur_op_name << " is not ready";
                    break;
                } else {
                    // 继续放回去，等待 ready
                    operators_queue.push_back(cur_op);
                }
            }

            // 当前算子可以执行

            const std::vector<std::shared_ptr<RuntimeOperand>>& input_operands = cur_op->input_operands_seq;
            // 所有输入的 operands 指针
            
            std::vector<std::shared_ptr<ftensor>> layer_input_datas; // 输入的数据

            for (const auto& input_operand_data : input_operands) {
                // 可能输入有多个操作数 - add，mul
                for (const auto& input_data : input_operand_data->datas) {
                    layer_input_datas.push_back(input_data);
                } 
            }

            CHECK(!layer_input_datas.empty()) << cur_op->name <<  " Layer input data is empty";
            CHECK(cur_op->output_operand != nullptr && !cur_op->output_operand->datas.empty()) << cur_op->name << " Layer output data is empty";

            // 开始当前算子计算
            cur_op->layer->Forward(layer_input_datas, cur_op->output_operand->datas);

            PassOutputToNext(cur_op, operators_queue, cur_op->output_operand->datas);
        }

        for (const auto& op : this->operators_) {
            op->meet_time = 0;
        }

        const auto& output_op_input_operand = output_op->input_operands.begin();
        // 输出算子的输入就是最终计算图的推理结果

        const auto& output_operand = output_op_input_operand->second;

        return output_operand->datas;
    } 

}

void RuntimeGraph::PassOutputToNext(const std::shared_ptr<RuntimeOperator>& cur_op, std::deque<std::shared_ptr<RuntimeOperator>>& operators_queue, std::vector<std::shared_ptr<Tensor<float>>> layer_output_datas) {
    // 输出节点列表
    const auto& next_ops = cur_op->output_operators;
    
    // 遍历所有的输出节点
    for (const auto& next_op : next_ops) {
        const auto& next_runtime_op = next_op.second; // 算子
        const auto& next_op_input_operands = next_runtime_op->input_operands; // 所有输入节点

        if (next_op_input_operands.find(cur_op->name) != next_op_input_operands.end()) {
            // 确实是下一个节点的输入,取出分配好的空间装载数据
            std::vector<std::shared_ptr<ftensor>>& next_input_datas = next_op_input_operands.at(cur_op->name)->datas;

            // 装载一个batch的数据
            for (uint32_t i = 0; i < next_input_datas.size(); ++i) {
                next_input_datas.at(i) = layer_output_datas.at(i);
            }

            next_runtime_op->meet_time += 1;
            if (std::find(operators_queue.begin(), operators_queue.end(), next_runtime_op) == operators_queue.end()) {
                if (CheckOperatorReady(next_runtime_op)) {
                    operators_queue.push_back(next_runtime_op);
                }
            }
        }
    }
    
}


bool RuntimeGraph::CheckOperatorReady(const std::shared_ptr<RuntimeOperator>& op) {
    CHECK(op != nullptr);
    CHECK(op->meet_time <= op->input_operands.size());
    if (op->meet_time == op->input_operands.size()) {
        return true;
    } else {
        return false;
    }
}

}
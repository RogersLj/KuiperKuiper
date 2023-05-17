# KuiperKuiper

inference

## first 
测试安装库，以及项目主 CMakeLists.txt 和测试的 CMakeLists.txt 编写

## second
主要的 data 数据结构
tensor.cpp
tensor_utils.cpp
load_data.cpp
测试通过

## third
添加 Operator 类
Operator 类包括算子的属性
例如relu有threshold，conv有kernel_size, stride, padding等

Layer 类包含算子的实际计算
属性和计算分离
需要通过Operator初始化

每一个算子，名字相同，但是有不一样的属性
也就是在一个计算图里，op不一样，但是op的计算是固定的
有多个conv算子，每个算子属性不一样，也就是需要构造多个Operator
但是每个conv调用的计算forward是一样的，因此只需要初始化构造一个layer
（注册表是这个？）

### 算子注册
首先根据计算图里算子节点的名字字符串，自动化初始化每个算子

注册表
typedef std::map<OpType, Creator> CreateRegistry;

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

每一个算子，名字相同，但是有不一样的属性，属性存在operator里
也就是在一个计算图里，op不一样，但是op的计算是固定的
有多个conv算子，每个算子属性不一样，也就是需要构造多个Operator
但是每个conv调用的计算forward是一样的，因此只需要初始化构造一个layer

## fourth 
算子注册
首先根据计算图里算子节点的名字字符串，自动化初始化每个算子

注册表
typedef std::map<OpType, Creator> CreateRegistry;

key是OpType，value是初始化的layer
如果没有在注册表里，第一次就会进行注册
之后面对相同的OpType，只需要查找注册表即可得到对应forward函数

- 工厂模式
可以根据不同的类型创建不同的类型对象
这里是根据不同的OpType创建不同的Layer

- 单例模式
它确保一个类只有一个实例，并提供对该实例的全局访问点。
一般用static关键字实现，这里注册表在全局只需要初始化一次

静态函数只属于类,不属于任何对象。静态函数只能访问静态成员变量和其他静态函数。

算子在定义的时候就会自动注册

当调用`LayerRegisterer::CreateLayer(const std::shared_ptr<Operator> &op)`函数的时候会从注册表里找到对应layer的creator并创建返回


## fifth
添加算子sigmoid和maxpooling

关键字explicit的作用是防止不必要的类型转换和隐式构造函数调用，从而提高代码的可读性和安全性。
构造函数 "= default" 语法，表示该构造函数应该由编译器自动生成。

## sixth

包装 pnnx 的 ir
pnnx::Operator -> RuntimeOperator
pnnx::Oprand -> RuntimeOpearnd
pnnx::Paramater -> RumtimeParameter
pnnx::Attribute -> RuntimeAttribute
pnnx::Graph -> RuntimeGraph 

## seventh

PNNX 里除了算子,还有一些表达式的运算节点
例如 ADD, MUL
这一类没有对应的算子,在PNNX的param里是已表达式的形式表示
因此需要将表达式节点解析为RuntimeGraph

并注册expression layer
回顾一下如何注册一个新的layer：
1. 首先定义op，op是layer的属性信息，就是除了计算之外的所有信息，都应该在op初始化或赋值。对于expression，op应该包括构建好的子计算图，也就是expression的计算图。使用expression进行初始化。

任何op都继承自operator类，operator在初始化的时候需要传入OpType，所以注册op时需要在op.hpp里添加新的optype

头文件定义后，去源文件定义每个函数的实现

2. 然后定义layer,layer层的构造函数，是通过op进行初始化。同时需要定义前向计算的函数forward。还需要有一个注册函数。这个函数在layer的函数实现进行layer的注册。


## eighth

卷子算子的添加

op -> layer
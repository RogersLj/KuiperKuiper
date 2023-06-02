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

卷积算子的添加

op -> layer

im2col优化


## ninth
优化tensor数据结构
原来的Tensor不能在逻辑上区分当前的张量是三维的、二维的还是一维的，因为实际的数据存储类arma::fcube总是一个三维数据。

用raw_shape记录张量的形状，三维的、二维的还是一维。
当raw_shape的长度为2时，说明当前的张量是二维的。当raw_shape的长度为1的时候，说明当前的张量是一维的。

Reshape的方式是列优先的，这是因为负责管理数据的armadillo::cube是一个列优先的容器。

构建图关系

前面只是成功转换了ir
没有将转换的ir构建可执行的计算图

构建图关系
找到输入和输出节点
输出节点的 input_oprands
就是最终模型的输出

对于算子的输入输出
可以提前分配好数据存放的空间

因为在没有做任何优化的情况下，我们假定是一个 batch 里的数据循环进行计算的
没有任何的并行，因此数据大小可以提前分配

算子的执行，采用广度优先的顺序
当一个算子的输入全部准备好，也就是上一个节点的输出拷贝到输入的的次数等于该节点的前驱节点个数
表示该节点已经准备好所需的输入操作数，因此可以加入执行队列中

计算图在推理过程中只需要构建一次,然后就是不同的输入得到最终的计算结果了
这个计算图就是静态图

init 完成对计算图ir的转换，build完成数据空间的分配

但是到现在都是空的,执行的算子是layer,但是此时还没有把runtimeOperator和layer映射起来


## tenth

最后需要完成前向推断,需要为每个RuntimeOperator创建对应的layer
但是之前创建layer是靠的Operator类,而经过包装后,创建layer时用到的Operator类的属性,现在可以直接从RuntimeOperator类里读取
因此需要对layer创建时的函数进行整体的修改,采用直接使用RuntimeOperator创建

同时对于参数和权重信息被封装在param和attrs里
现在也需要将其内容放到layer里

先从forward函数里看，如果对原有的代码进行重构
因为我们需要在forward函数里进行layer的计算

---

很重要的一点
因为之前op存储属性,layer进行计算
对应到现在,layer已经被封装进RuntimeOperator里
而属性信息,即参数和权重也被存放在RuntimeOperator的params和attrs里
因此,我们在初始化layer的时候,其实不需要另外使用一个op

对之前的代码进行改进,就是要通过RuntimeOperator初始化layer
同时将RuntimeOperator的信息直接传给layer层
当在图build函数的时候,在创建layer之前,已经初始化完所有的operators,在init函数里,
同时param和attrs信息也已经存入operator里,
因此在创建layer的时候,要用op里的参数和权重信息初始化layer
之前的这部分信息由op存储,而现在需要直接存储在layer里
因此代码需要修改的一部分是给layer添加一些属性信息(和之前op和layer,属性和计算分开有些不一样了)

### 重点:重写Layer类
- 对于所有的参数来说,一般分为​可训练参数和超参数,在常见的视觉网络里.例如resnet,yolov5中,除了卷积层和全连接层,其他都是超参数
- 而我们在推理阶段所用于计算的,一般都是训练得到的参数部分，卷积和全连接层的权重参数和偏置
- 所以基于基类layer,定义一个paramlayer,定义专门用于weight和bias的函数,方便继承类使用


之后在计算的时候,forward时其实就是调用RuntimeOperator里面的layer的forward函数进行计算

主要的改动就是将之前每个算子用operator类创建layer的函数改成
通过runtimeoperator类创建

--- 

修改后代码的运行流程是这样的：
1. 当整个框架编译完成的时候，就完成了算子的注册，注册表为算子的名字和算子layer的创建函数
2. 当前向推理时，会找到当前算子对应的layer创建函数，并传入当前RuntimeOperator，初始化layer信息，然后将初始化好的layer放进RuntimeOperator里。初始化的信息包括具体的参数和权重。这部分已经在RuntimeOperator类的params和attrs里。
3. 本来layer的构造函数由原本的Operator类进行构造。现在没有了中间的Operator类，因此首先需要修改我们layer的构造函数，构造参数会由RuntimeOperator的params和attrs信息进行构造。如果该层有weghts和bias，则继承与ParamLayer类，里面的weights和bias用于存储数据。如果是没有weights和bias的layer，则直接在继承类里保存相应信息。
4. 最后每个算子的creator函数，就是CreateInstance函数，通过传入的RuntimeOperator类返回当前初始化好的layer实例。该函数就是被放入注册表，在推断时查找的创建layer的函数。
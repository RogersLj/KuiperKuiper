#ifndef KUIPER_INFER_FACTORY_LAYER_FACTORY_HPP
#define KUIPER_INFER_FACTORY_LAYER_FACTORY_HPP

#include "ops/op.hpp"
#include "layer/layer.hpp"

namespace kuiper_infer {

// 注册表相关函数

class LayerRegister {

public:

    // 定义了一个函数指针类型Creator，参数为Operator对象
    // 返回值为指向Layer对象的指针
    // 多个op可以指向同一个Layer对象
    // Create就是每个Layer的创建函数
    // 根据不同layer的creator创建不同的layer层
    // 因为相同名字的op_type对于的layer都是一样的
    // 所以第一次遇到的算子需要加入注册表,之后见到直接查找表并创建layer层
    typedef std::shared_ptr<Layer> (*Creator) (const std::shared_ptr<Operator>& op);

    // 注册表，key是OpType,value是Creator
    // 查找表就可以找到对于算子的一个初始化方法
    // 第一次遇到的算子需要加入注册表,之后见到直接查找表并找到creator创建layer层
    typedef std::map<OpType, Creator> CreateRegistry;


    // 注册算子函数，根据算子的类型以及创建函数，加入注册表里
    static void RegisterCreator(OpType op_type, const Creator& creator);


    // 根据op的类型创建对应的Layer
    static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<Operator>& op);

    // 创建一个注册表，并返回引用
    // 全局只需要一个注册表，所以会看到用static生明的注册表变量
    static CreateRegistry& Registry();
};

/*
已知算子op_type,在注册表里注册的过程:
首先拿到op_type，
*/

class LayerRegisterWrapper {
public:
// 每定义完一个算子，就会加入注册表
// 根据(op_type, creator)
    LayerRegisterWrapper(OpType op_type, const LayerRegister::Creator& creator) {
        LayerRegister::RegisterCreator(op_type, creator);
    }
};

}
#endif
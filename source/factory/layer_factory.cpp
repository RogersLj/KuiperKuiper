#include "factory/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

void LayerRegister::RegisterCreator(OpType op_type, const Creator& creator) {
    CHECK(creator != nullptr);
    // 初始化注册表，如果第一次创建，则新建注册表并初始化
    // 如果已经初始化了注册表，则不初始化，直接返回 static 实例
    CreateRegistry &registry = Registry();

    // 因为算子在定义时就被注册，只会注册一次
    CHECK_EQ(registry.count(op_type), 0) << "Layer type: " << int(op_type)
    << " has already been registered.";

    // 每个op_type都指向一个layer的creator
    registry.insert({op_type, creator});
}


// 注册表的初始化，返回引用
LayerRegister::CreateRegistry& LayerRegister::Registry() {
    // static 只会初始化一次
    static CreateRegistry *kRegistry = new CreateRegistry();
    CHECK(kRegistry != nullptr);
    return *kRegistry;
}

// 实际上只需要根据op调用CreateLayer函数，就可以得到对应的layer
std::shared_ptr<Layer> LayerRegister::CreateLayer(
    const std::shared_ptr<Operator>& op) {
    CreateRegistry &registry = Registry();
    const OpType op_type = op->op_type_;

    LOG_IF(FATAL, registry.count(op_type) <= 0) << "Can not find the layer type: " << int(op_type);        

    // 根据op_type获取对应的creator
    const auto& creator = registry.find(op_type)->second;

    LOG_IF(FATAL, !creator) << "Layer creator is empty!";

    std::shared_ptr<Layer> layer = creator(op);
    LOG_IF(FATAL, !layer) << "Layer init failed!";
    return layer;
}

}
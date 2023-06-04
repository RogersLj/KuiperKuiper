#include "layer/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

void LayerRegister::RegisterCreator(const std::string& layer_type , const Creator& creator) {
    CHECK(creator != nullptr);
    // 初始化注册表，如果第一次创建，则新建注册表并初始化
    // 如果已经初始化了注册表，则不初始化，直接返回 static 实例
    CreateRegistry &registry = Registry();

    // 因为算子在定义时就被注册，只会注册一次 - 注册就是得到creator,不同层只是参数不一样,creator都是一样的,放在注册表里
    CHECK_EQ(registry.count(layer_type), 0) << "Layer type: " << layer_type
    << " has already been registered.";

    // 每个op_type都指向一个layer的creator
    registry.insert({layer_type, creator});
}


// 注册表的初始化，返回引用
LayerRegister::CreateRegistry& LayerRegister::Registry() {
    // static 只会初始化一次
    static CreateRegistry *kRegistry = new CreateRegistry();
    CHECK(kRegistry != nullptr);
    return *kRegistry;
}

std::shared_ptr<Layer> LayerRegister::CreateLayer(const std::shared_ptr<RuntimeOperator>& op) {
    CreateRegistry &registry = Registry();
    const std::string& layer_type = op->type;
    LOG_IF(FATAL, registry.count(layer_type) <= 0) << "Can not find the layer type: " << layer_type;

    const auto& creator = registry.find(layer_type)->second;
    LOG_IF(FATAL, !creator) << "Layer creator is empty!";

    std::shared_ptr<Layer> layer;
    creator(op, layer);

    if (!layer) {
        LOG(FATAL) << "Layer init failed!";
    } 

    return layer;
}

}
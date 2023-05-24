#ifndef KUIPER_INFER_RUNTIME_ATTR_HPP
#define KUIPER_INFER_RUNTIME_ATTR_HPP

// attribute - 算子的属性信息

#include <vector>
#include <glog/logging.h>
#include "status_code.hpp"
#include "runtime_datatype.hpp"

namespace kuiper_infer {
    
struct RuntimeAttribute {
    
    // 用字节存储，利于序列化和反序列化
    std::vector<char> weight_data; // 权重参数
    std::vector<int> shape; // 参数的具体形状

    RuntimeDataType type = RuntimeDataType::kTypeUnknown;

    // 根据算子不同类型，返回相应的权重参数
    template<typename T>
    std::vector<T> get();
};

template<typename T>
std::vector<T> RuntimeAttribute::get() {
    
    CHECK(!weight_data.empty());

    CHECK(type != RuntimeDataType::kTypeUnknown);

    std::vector<T> weights;

    switch (type) {
        case RuntimeDataType::kTypeFloat32: {
            const bool is_float = std::is_same<T, float>::value;
            CHECK_EQ(is_float, true);
            const uint32_t float_size = sizeof(float);
            CHECK_EQ(weight_data.size() % float_size, 0);
            for (uint32_t i = 0; i < weight_data.size() / float_size; ++i) {
                float weight = *((float*)weight_data.data() + i);
                weights.push_back(weight);
            }
            break;
        }

        default: {
            LOG(FATAL) << "Unknown weight data type";
        }
    }

    return weights;
}



}


#endif
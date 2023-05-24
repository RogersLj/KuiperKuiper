#ifndef KUIPER_INFER_RUNTIME_PARAM_HPP
#define KUIPER_INFER_RUNTIME_PARAM_HPP

#include "status_code.hpp"
#include <string>
#include <vector>

namespace kuiper_infer {
    
/*
参数的类型 - bool, int, float, string, int array, float array, string array
enum class RuntimeParameterType {
  kParameterUnknown = 0,
  kParameterBool = 1,
  kParameterInt = 2,

  kParameterFloat = 3,
  kParameterString = 4,
  kParameterIntArray = 5,
  kParameterFloatArray = 6,
  kParameterStringArray = 7,
};
*/

// 参数 - RuntimeParameterType type 和 value
struct RuntimeParameter {
    virtual ~RuntimeParameter() = default;

    explicit RuntimeParameter(RuntimeParameterType type = RuntimeParameterType::kParameterUnknown) : type(type) {};

    RuntimeParameterType type = RuntimeParameterType::kParameterUnknown;
};

struct RuntimeParameterInt : public RuntimeParameter {
    RuntimeParameterInt() : RuntimeParameter(RuntimeParameterType::kParameterInt) {};

    int value = 0;
};

struct RuntimeParameterFloat : public RuntimeParameter {
    RuntimeParameterFloat() : RuntimeParameter(RuntimeParameterType::kParameterFloat) {};

    float value = 0.f;
};

struct RuntimeParameterString : public RuntimeParameter {
    RuntimeParameterString() : RuntimeParameter(RuntimeParameterType::kParameterString) {};

    std::string value;
};

struct RuntimeParameterIntArray : public RuntimeParameter {
    RuntimeParameterIntArray() : RuntimeParameter(RuntimeParameterType::kParameterIntArray) {};

    std::vector<int> value;
};

struct RuntimeParameterFloatArray : public RuntimeParameter {
    RuntimeParameterFloatArray() : RuntimeParameter(RuntimeParameterType::kParameterFloatArray) {};

    std::vector<float> value;
};

struct RuntimeParameterStringArray : public RuntimeParameter {
    RuntimeParameterStringArray() : RuntimeParameter(RuntimeParameterType::kParameterStringArray) {};

    std::vector<std::string> value;
};

struct RuntimeParameterBool : public RuntimeParameter {
    RuntimeParameterBool() : RuntimeParameter(RuntimeParameterType::kParameterBool) {};

    bool value = false;
};


}


#endif

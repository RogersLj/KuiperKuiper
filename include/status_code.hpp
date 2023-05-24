#ifndef KUIPER_INFER_STATUS_CODE_HPP
#define KUIPER_INFER_STATUS_CODE_HPP

namespace kuiper_infer {
    
// 参数类型
// 对应pnnx的参数定义
// 0-null,1-bool,2-integer,3-float,4-string
// 5-vector of integers,6-vector of floats, 7-vector of strings
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



}

#endif
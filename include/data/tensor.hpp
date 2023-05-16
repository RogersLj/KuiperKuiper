#ifndef KUIPER_INFER_DATA_H
#define KUIPER_INFER_DATA_H

#include <memory>
#include <vector>
#include <armadillo>

namespace kuiper_infer {

// 模板类
template <typename T>
class Tensor {
};

// 量化 8bit 待实现
template <>
class Tensor<uint8_t> {    
};


// 特化 float 模板
template <>
class Tensor<float> {
public:
    explicit Tensor() = default;

    // 构造函数 - CHW
    explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

    explicit Tensor(const std::vector<uint32_t>& shape);

    // 构造函数 - 用已知 Tensor 赋值
    Tensor(const Tensor& tensor);

    Tensor(Tensor&& tensor) noexcept;

    // 运算符重载
    Tensor<float>& operator=(const Tensor<float>& tensor);

    Tensor<float>& operator=(Tensor<float>&& tensor) noexcept;

    // 返回 channels， rows， cols
    uint32_t channels() const;

    uint32_t rows() const;

    uint32_t cols() const;

    // 返回 张量元素个数
    uint32_t size() const;

    // 用 arma::fcube 进行数据赋值
    void set_data(const arma::fcube& data);

    // 判断是否为空
    bool empty() const;

    // 根据 index 返回张量连续的元素值
    // const 成员函数不会修改对象的数据
    float index(uint32_t offset) const;

    // 返回引用，即可修改
    // t.index(offset) = value
    float &index(uint32_t offset);

    // 返回张量形状 - CHW
    std::vector<uint32_t> shape() const;

    // 返回张量实际用 arma::fcube 存储的形状 - HWC
    const std::vector<uint32_t>& raw_shape() const;


    // 返回张量的原始数据
    arma::fcube& data();

    const arma::fcube& data() const;

    // 返回某一通道的张量
    arma::fmat& slice(uint32_t channel);

    const arma::fmat& slice(uint32_t channel) const;

    // 返回某一CHW位置的张量
    float at(uint32_t channel, uint32_t row, uint32_t col) const;

    float &at(uint32_t channel, uint32_t row, uint32_t col);

    // 填充 padding
    void Padding(const std::vector<uint32_t>& pads, float padding_value);

    // 初始化张量值
    void Fill(float value);

    // 用数组初始化所有值
    void Fill(const std::vector<float>& values, bool row_major = true);

    // 返回张量所有值
    std::vector<float> values(bool row_major = true);

    // 初始化
    void Ones();

    void Zeros();

    void Rand();

    // 打印张量
    void Show();
 
    // 对张量进行 reshape 根据列主序或行主序
    void Reshape(const std::vector<uint32_t>& shape, bool row_major = false);

    // 展平张量 - 默认 fcube 是列主序的
    void Flatten(bool row_major = false);

    // 根据函数对张量元素进行过滤
    void Transform(const std::function<float(float)>& filter);

    // 返回张量的原始指针
    const float* raw_ptr() const;
 
private:
    std::vector<uint32_t> raw_shape_;  // 张量数据的实际尺寸大小
    arma::fcube data_;                  // 张量数据

};

using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;

} // namespace kuiper_infer

#endif  // KUIPER_INFER_DATA_HPP
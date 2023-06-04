#include "data/tensor.hpp"
#include <glog/logging.h>
#include <memory>
#include <numeric>

namespace kuiper_infer {

// 默认所有张量都是三维的,没有的维度也需要用1表示,因为arma底层是这样初始化的
// raw_shape_ 存储的是最底层的逻辑形状
// 因为fcube按照HWC的形状来分,所以这里的raw_shape_和data_的shape不一样
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    // fcube 初始化 - HWC - 内存排布 列主序
    data_ = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        // 一维张量 - 看起来是个行向量
        this->raw_shape_ = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        // 二维张量
        this->raw_shape_ = std::vector<uint32_t>{rows, cols};
    } else {
        // 三维张量
        this->raw_shape_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

// 习惯用CHW初始化 - 允许直接​初始化一维或二维张量
Tensor<float>::Tensor(const std::vector<uint32_t>& shape) {
    CHECK(!shape.empty() && shape.size() <= 3);
    
    uint32_t dim = shape.size(); // 维度
    uint32_t remains = 3 - dim; // 要补全的维度

    std::vector<uint32_t> new_shape(3, 1); // 默认形状是三维,然后将维度真实值补全

    std::copy(shape.begin(), shape.end(), new_shape.begin() + remains);
    // 从后往前补全 
    
    uint32_t channels = new_shape.at(0);
    uint32_t rows = new_shape.at(1);
    uint32_t cols = new_shape.at(2);

    data_ = arma::fcube(rows, cols, channels);

    if (channels == 1 && rows == 1) {
        this->raw_shape_ = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        this->raw_shape_ = std::vector<uint32_t>{rows, cols};
    } else {
        this->raw_shape_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

Tensor<float>::Tensor(const Tensor<float>& tensor) {
    // 传入 tensor 引用，&tensor 实际 tensor 地址
    if (this != &tensor) {
        this->data_ = std::move(tensor.data_);
        this->raw_shape_ = tensor.raw_shape_;
    }
}

Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
    if (this != &tensor) {
        this->data_ = std::move(tensor.data_);
        this->raw_shape_ = tensor.raw_shape_;
    }
}

Tensor<float>& Tensor<float>::operator=(const Tensor<float>& tensor) {
    if (this != &tensor) {
        this->data_ = std::move(tensor.data_);
        this->raw_shape_ = tensor.raw_shape_;
    }
    return *this;
}

Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
    if (this != &tensor) {
        this->data_ = std::move(tensor.data_);
        this->raw_shape_ = tensor.raw_shape_;
    }
    return *this;
}

uint32_t Tensor<float>::channels() const {
    CHECK(!this->data_.empty());
    return this->data_.n_slices;
}

uint32_t Tensor<float>::rows() const {
    CHECK(!this->data_.empty());
    return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
    CHECK(!this->data_.empty());
    return this->data_.n_cols;
}

uint32_t Tensor<float>::size() const {
    CHECK(!this->data_.empty());
    return this->data_.size();
}


void Tensor<float>::set_data(const arma::fcube& data) {
    CHECK(data.n_rows == this->data_.n_rows)
      << data.n_rows << " != " << this->data_.n_rows;
    CHECK(data.n_cols == this->data_.n_cols)
      << data.n_cols << " != " << this->data_.n_cols;
    CHECK(data.n_slices == this->data_.n_slices)
      << data.n_slices << " != " << this->data_.n_slices;
    
    this->data_ = data;
}

bool Tensor<float>::empty() const {
    return this->data_.empty();
}


// column-major
float Tensor<float>::index(uint32_t offset) const {
    CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
    return this->data_.at(offset);
}

float& Tensor<float>::index(uint32_t offset) {
    CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
    return this->data_.at(offset);
}


// t.shape() - CHW - 不是逻辑形状,为1的维度也同样返回
std::vector<uint32_t> Tensor<float>::shape() const {
    CHECK(!this->data_.empty());
    return {this->channels(), this->rows(), this->cols()};
}

arma::fcube& Tensor<float>::data() { 
    return this->data_; 
}

const arma::fcube& Tensor<float>::data() const { 
    return this->data_; 
}

arma::fmat& Tensor<float>::slice(uint32_t channel) {
    CHECK_LT(channel, this->channels());
    return this->data_.slice(channel);
}

const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
    CHECK_LT(channel, this->channels());
    return this->data_.slice(channel);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
    CHECK_LT(row, this->rows());
    CHECK_LT(col, this->cols());
    CHECK_LT(channel, this->channels());
    return this->data_.at(row, col, channel);   
}

float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
    CHECK_LT(row, this->rows());
    CHECK_LT(col, this->cols());
    CHECK_LT(channel, this->channels());
    return this->data_.at(row, col, channel);
}

//(padding_left,padding_right,padding_top,padding_bottom)
void Tensor<float>::Padding(const std::vector<uint32_t>& pads, float padding_value) {
    CHECK(!this->data_.empty());
    CHECK_EQ(pads.size(), 4);

    uint32_t padding_left = pads.at(0);
    uint32_t padding_right = pads.at(1);
    uint32_t padding_top = pads.at(2);
    uint32_t padding_bottom = pads.at(3);

    arma::fcube new_data(this->rows() + padding_top + padding_bottom, this->cols() + padding_left + padding_right, this->channels());

    new_data.fill(padding_value);

    // Q.subcube( first_row, first_col, first_slice, last_row, last_col, last_slice )
    new_data.subcube(padding_top, padding_left, 0, 
                    padding_top + this->rows() - 1, padding_left + this->cols() - 1, new_data.n_slices - 1) = this->data_;

    this->data_ = std::move(new_data);
}

void Tensor<float>::Fill(float value) {
    CHECK(!this->data_.empty());
    this->data_.fill(value);
}

void Tensor<float>::Fill(const std::vector<float>& values, bool row_major) {
    CHECK(!this->data_.empty());
    
    const uint32_t total_elems = this->data_.size();
    CHECK_EQ(values.size(), total_elems);

    if (row_major) {
        const uint32_t channels = this->channels();
        const uint32_t rows = this->rows();
        const uint32_t cols = this->cols();
        const uint32_t planes = rows * cols;

        for (uint32_t i = 0; i < channels; ++i) {
            auto& cur_channel = this->data_.slice(i);
            const arma::fmat& channel_data = arma::fmat(values.data() + i * planes, cols, rows);
            cur_channel = channel_data.t(); // 转置
        }
    } else {
        std::copy(values.begin(), values.end(), this->data_.memptr());
    }
}

void Tensor<float>::Show() {
    for (uint32_t i = 0; i < this->channels(); ++i) {
        LOG(INFO) << "Channel: " << i;
        LOG(INFO) << "\n" << this->data_.slice(i);
    }
}

void Tensor<float>::Rand() {
    CHECK(!this->data_.empty());
    this->data_.randn();
}

void Tensor<float>::Ones() {
    CHECK(!this->data_.empty());
    this->Fill(1.f);
}

void Tensor<float>::Zeros() {
    CHECK(!this->data_.empty());
    this->Fill(0.f);
}

void Tensor<float>::Flatten(bool row_major) {
  CHECK(!this->data_.empty());
  const uint32_t size = this->data_.size();
  this->Reshape({size}, row_major);
}

void Tensor<float>::Transform(const std::function<float(float)>& filter) {
  CHECK(!this->data_.empty());
  this->data_.transform(filter);
}

void Tensor<float>::Reshape(const std::vector<uint32_t>& shape, bool row_major) {
    CHECK(!this->data_.empty());
    CHECK(!shape.empty());

    const uint32_t origin_size = this->size();
    const uint32_t current_size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    CHECK(shape.size() <= 3);
    CHECK(current_size == origin_size);

    std::vector<float> values;
    if (row_major) {
        values = this->Values(true); // 把数据按行展开存放在values中
    }
    if (shape.size() == 1) {
        // fcube 默认是 HWC - 对data_ reshape
        // 传入的是 CHW
        this->data_.reshape(1, shape.at(0), 1);
        this->raw_shape_ = {shape.at(0)};
    } else if (shape.size() == 2) {
        this->data_.reshape(shape.at(0), shape.at(1), 1);
        this->raw_shape_ = {shape.at(1), shape.at(2)};
    } else {
        this->data_.reshape(shape.at(1), shape.at(2), shape.at(0));
        this->raw_shape_ = {shape.at(0), shape.at(1), shape.at(2)};
    }

    if (row_major) {
        this->Fill(values, true);
    }
}

const std::vector<uint32_t>& Tensor<float>::raw_shape() const {
    CHECK(!this->raw_shape_.empty());
    CHECK_LE(this->raw_shape_.size(), 3);
    CHECK_GE(this->raw_shape_.size(), 1);
    return this->raw_shape_;
}

const float* Tensor<float>::raw_ptr() const {
  CHECK(!this->data_.empty());
  return this->data_.memptr();
}

std::vector<float> Tensor<float>::Values(bool row_major) {
    CHECK_EQ(this->data_.empty(), false);
    std::vector<float> values(this->data_.size());

    if (!row_major) {
        std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
              values.begin());
  } else {
        uint32_t index = 0;
        for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
            const arma::fmat& channel_data = this->data_.slice(c).t();
            std::copy(channel_data.begin(), channel_data.end(), values.begin() + index);
            index += channel_data.size();
        }
        CHECK_EQ(index, values.size());
  }
  return values;
}


} // namespace kuiper_infer
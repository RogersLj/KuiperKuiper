#include "layer/conv_layer.hpp"
#include "ops/conv_op.hpp"
#include "data/tensor_util.hpp"
#include "factory/layer_factory.hpp"
#include <glog/logging.h>


namespace kuiper_infer {

ConvLayer::ConvLayer(const std::shared_ptr<Operator> &op) : Layer("ConvLayer") {
    CHECK(op != nullptr && op->op_type_ == OpType::kOperatorConv);
    ConvOp* conv_op = dynamic_cast<ConvOp*>(op.get());

    CHECK(conv_op != nullptr) << "Conv op is empty!";
    this->op_ = std::make_unique<ConvOp>(*conv_op);
}    

void ConvLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    CHECK(this->op_ != nullptr);
    CHECK(this->op_->op_type_ == OpType::kOperatorConv);
    CHECK(!inputs.empty());

    auto stride = this->op_->get_stride();
    auto padding = this->op_->get_padding();

    const uint32_t batch_size = inputs.size();

    const uint32_t padding_h = padding.first;
    const uint32_t padding_w = padding.second;
    const uint32_t stride_h = stride.first;
    const uint32_t stride_w = stride.second;
    const uint32_t groups = this->op_->get_groups();

    const std::vector<sftensor>& weights = this->op_->get_weights();
    CHECK(!weights.empty());


    for (uint32_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<float>> &input_data = TensorClone(inputs.at(i));

        CHECK(input_data != nullptr);

        if (padding_w != 0 || padding_h != 0)
            input_data->Padding({padding_w, padding_w, padding_h, padding_h} ,0);

        // batch里的一个输入

        const uint32_t input_h = inputs.at(0)->rows();
        const uint32_t input_w = inputs.at(0)->cols();
        const uint32_t input_c = inputs.at(0)->channels();
        
        const uint32_t output_c = weights.size(); // 卷积核个数
        const uint32_t kernel_h = weights.at(0)->rows();
        const uint32_t kernel_w = weights.at(0)->cols();
        const uint32_t kernel_c = weights.at(0)->channels();
        const uint32_t output_h = std::uint32_t(std::floor((input_h  - kernel_h) / stride_h)) + 1;
        const uint32_t output_w = std::uint32_t(std::floor((input_w  - kernel_w) / stride_w)) + 1;

        if (groups != 1) {
            CHECK(input_c % groups == 0);
            CHECK(output_c % groups == 0);
            CHECK(input_c / groups == kernel_c);
        }

        // im2col
        // 将输入转换为一个列为kernel大小，行为输出大小的张量
        const uint32_t kernel_size = kernel_h * kernel_w; // 一个通道的kernel大小
        // 一个卷积操作的输出大小
        const uint32_t output_size = output_h * output_w;

        // outputs - (batch_size, output_channels, output_h, output_w)
        
        // 当前batch的输出
        std::shared_ptr<ftensor> output_data = std::make_shared<ftensor>(output_c, output_h, output_w);

        uint32_t kernels_per_group = output_c / groups;
        for (uint32_t g = 0; g < groups; ++g) {
            // 拼接一个group的kernel和一个group的输入
            // 存放展开后的kernel，大小为一个group的kernels
            std::vector<arma::fmat> kernel_matrix(kernels_per_group);
            arma::fmat kernel_flatten(1, kernel_size * kernel_c);
            // 一个展开的kernel大小，kernel的channels是一个group的输入channels

            for (uint32_t k = 0; k < kernels_per_group; ++k) {
                // 展开一个group的kernel
                // 拿到当前的kernel
                const std::shared_ptr<Tensor<float>> &kernel = weights.at(k + kernels_per_group * g);
                
                // 按通道展开
                for (uint32_t ic = 0; ic < kernel_c; ++ic) {
                    memcpy(kernel_flatten.memptr() + ic * kernel_size, kernel->slice(ic).memptr(), kernel_size * sizeof(float));
                }

                kernel_matrix.at(k) = kernel_flatten;
                // 一个group的展开kernel_matrix
                // (kernel_per_group, kernel_size * kernel_c)
            }

            // im2col输入矩阵 - (kernel_size * kernel_c, output_size)
            // 最终输出 kernel_matrix @ input_matrix - (kernel_per_group, output_size) - 每一个卷积得到一个output_channel
            arma::fmat input_matrix(kernel_size * kernel_c, output_size);
            
            // kernel_c == input_channels_per_group
            
            for (uint32_t ic = 0; ic < kernel_c; ++ic) {
                const arma::fmat &input_channel = input_data->slice(g * kernel_c + ic);
                // 取一个通道的输入
                uint32_t cur_col = 0; // 遍历到的input_matrix的列
                for (uint32_t irow = 0; irow < input_h - kernel_h + 1; irow += stride_h)
                    for (uint32_t jcol = 0; jcol < input_w - kernel_w + 1; jcol += stride_w) {
                        // 当前channel应该放在input_matrix的cur_col的具体位置
                        float* input_matrix_c_colptr = input_matrix.colptr(cur_col) + ic * kernel_size;
                        cur_col += 1;

                        // 列优先处理窗口数据
                        for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                            const float *region_ptr = input_channel.colptr(jcol + kw) + irow;
                            // 当前区域左上角地址
                            memcpy(input_matrix_c_colptr, region_ptr, kernel_h * sizeof(float));

                            input_matrix_c_colptr += kernel_h;
                        }
                    }
            }

            // 还是在一个group里
            LOG(INFO) << "input展开后: " << "\n" << input_matrix;

            // std::shared_ptr<ftensor> output_data = outputs.at(i);
            // // 初始化当前batch的输出
            // // 因为是在group里，只需要初始化一次
            // if (output_data == nullptr || output_data->empty()) {
            //     output_data = std::make_shared<ftensor>(output_c, output_h, output_w);
            //     outputs.at(i) = output_data;
            // }

            // std::vector<arma::fmat> outputs_matrix(kernels_per_group);

            for (uint32_t k = 0; k < kernels_per_group; k ++) {
                LOG(INFO) << "这是第" << k << "个卷积\n" << kernel_matrix.at(k); // 拿出第一个展开后的卷积核
                LOG(INFO) << "\n" << input_matrix; // 拿出展开后的输入特征图

                const arma::fmat& output = kernel_matrix.at(k) * input_matrix;

                LOG(INFO) << "当前卷积结果：\n" << output; // 当前卷积算子的输出

                // outputs_matrix.at(k) = output;

                arma::fmat& output_slice = output_data->slice(g * kernels_per_group + k);

                arma::fmat output_biased = output;
                output_biased.reshape(output_h, output_w);

                if (this->op_->get_has_bias()) {
                    LOG(INFO) << "当前有bias：\n";
                    std::vector<std::shared_ptr<Tensor<float>>> bias = this->op_->get_bias();
                    
                    LOG(INFO) << "reshape后的输出：\n" << output;
                    LOG(INFO) << "需要加到output上的bias：\n" << bias.at(k + kernels_per_group * g);
                    
                    output_biased += bias.at(k + kernels_per_group * g)->index(0);
                } 
                
                output_slice = output_biased;

                // output_data->slice(g * kernels_per_group + k) = std::move(output);
                
            }

        }
        outputs.at(i) = std::move(output_data);

    }
}
}
#include <glog/logging.h>
#include "ops/maxpooling_op.hpp"
#include "layer/maxpooling_layer.hpp"
#include "data/tensor_util.hpp"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    
MaxPoolingLayer::MaxPoolingLayer(const std::shared_ptr<Operator> &op) : Layer("MaxPoolingLayer") {
    CHECK(op->op_type_ == OpType::kOperatorMaxPooling)
        << "Operator " << int(op->op_type_) << " is not MaxPoolingOp!";

    MaxPoolingOp* maxpooling_op = dynamic_cast<MaxPoolingOp*>(op.get());

    CHECK(maxpooling_op != nullptr) << "MaxPooling op is empty!";

    this->op_ = std::make_unique<MaxPoolingOp>(*maxpooling_op);
}

void MaxPoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    CHECK(this->op_ != nullptr);
    CHECK(this->op_->op_type_ == OpType::kOperatorMaxPooling);
    CHECK(!inputs.empty());

    auto kernel_size = this->op_->get_kernel_size();
    auto stride = this->op_->get_stride();
    auto padding = this->op_->get_padding();


    const uint32_t padding_h = padding.first;
    const uint32_t padding_w = padding.second;
    const uint32_t kernel_h = kernel_size.first;
    const uint32_t kernel_w = kernel_size.second;
    const uint32_t stride_h = stride.first;
    const uint32_t stride_w = stride.second;

    const uint32_t batch_size = inputs.size();

    for (uint32_t i = 0; i < batch_size; ++i) {
        
        const std::shared_ptr<Tensor<float>> &input_data = TensorClone(inputs.at(i));

        //(padding_left,padding_right,padding_top,padding_bottom)
        // padding_h - padding_top,padding_bottom
        // padding_w - padding_left,padding_right
        input_data->Padding({padding_w, padding_w, padding_h, padding_h} ,std::numeric_limits<float>::lowest());

        const uint32_t input_h = input_data->rows();
        const uint32_t input_w = input_data->cols();
        const uint32_t input_c = input_data->channels();

        const uint32_t output_c = input_c;

        const uint32_t output_h = (input_h - kernel_h) / stride_h + 1;
        const uint32_t output_w = (input_w - kernel_w) / stride_w + 1;

        std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(output_c, output_h, output_w);


        for (uint32_t c = 0; c < input_c; ++c) {
            const arma::fmat& input_channel = input_data->slice(c); 
            arma::fmat& output_channel = output->slice(c);

            for (uint32_t h = 0; h + kernel_h <= input_h; h += stride_h) {
                for (uint32_t w = 0; w + kernel_w <= input_w; w += stride_w) {
                    const arma::fmat& sub = input_channel.submat(h, w, h + kernel_h - 1, w + kernel_w - 1);
                    output_channel.at(int(h / stride_h), int(w / stride_w)) = sub.max();
                }
            }
        }
        outputs.push_back(output);
    }

}

std::shared_ptr<Layer> MaxPoolingLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
    CHECK(op->op_type_ == OpType::kOperatorMaxPooling);
    return std::make_shared<MaxPoolingLayer>(op);
}

// 注册池化层
LayerRegisterWrapper kMaxPoolingLayer(OpType::kOperatorMaxPooling, MaxPoolingLayer::CreateInstance);

}
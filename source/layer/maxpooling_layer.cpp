#include <glog/logging.h>
#include "layer/maxpooling_layer.hpp"
#include "data/tensor_util.hpp"
#include "factory/layer_factory.hpp"

namespace kuiper_infer {
    
MaxPoolingLayer::MaxPoolingLayer(Shape kernel_size, Shape stride, Shape padding) : Layer("MaxPoolingLayer"), kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    
}

void MaxPoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    CHECK(!inputs.empty()) << "inputs is empty";

    const uint32_t padding_h = this->padding_.first;
    const uint32_t padding_w = this->padding_.second;
    const uint32_t kernel_h = this->kernel_size_.first;
    const uint32_t kernel_w = this->kernel_size_.second;
    const uint32_t stride_h = this->stride_.first;
    const uint32_t stride_w = this->stride_.second;

    const uint32_t batch_size = inputs.size();

    for (uint32_t i = 0; i < batch_size; ++i) {
        
        const std::shared_ptr<Tensor<float>> &input_data = TensorClone(inputs.at(i));

        //(padding_left,padding_right,padding_top,padding_bottom)
        // padding_h - padding_top,padding_bottom
        // padding_w - padding_left,padding_right
        
        if (padding_w != 0 || padding_h != 0) {
            input_data->Padding({padding_w, padding_w, padding_h, padding_h} ,std::numeric_limits<float>::lowest());
        }

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
        outputs.at(i) = output;
    }

}

void MaxPoolingLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer>& maxpooling_layer) {
    CHECK(op != nullptr) << "op is nullptr";

    const std::map<std::string, RuntimeParameter*>& params = op->params;

    const auto& kernel_size = dynamic_cast<RuntimeParameterIntArray*>(params.at("kernel_size"));
    const auto& stride = dynamic_cast<RuntimeParameterIntArray*>(params.at("stride"));
    const auto& padding = dynamic_cast<RuntimeParameterIntArray*>(params.at("padding"));

    const auto& kernel_size_v = kernel_size->value;
    const auto& stride_v = stride->value;
    const auto& padding_v = padding->value;

    Shape k = {kernel_size_v.at(0), kernel_size_v.at(1)};
    Shape s = {stride_v.at(0), stride_v.at(1)};
    Shape p = {padding_v.at(0), padding_v.at(1)};

    maxpooling_layer = std::make_shared<MaxPoolingLayer>(k, s, p);

}

// 注册池化层
LayerRegisterWrapper kMaxPoolingLayer("nn.MaxPool2d", MaxPoolingLayer::CreateInstance);

}
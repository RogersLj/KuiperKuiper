#include "./adaptive_avgpooling.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
    

AdaptiveAvgPoolingLayer::AdaptiveAvgPoolingLayer(uint32_t output_h, uint32_t output_w) : Layer("AdaptiveAvgPooling"), output_h_(output_h), output_w_(output_w) {


}


void AdaptiveAvgPoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs, std::vector<std::shared_ptr<Tensor<float>>>& outputs) {

    const uint32_t batch_size = inputs.size();

    for (uint32_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<float>> input_data = inputs.at(i);

        const uint32_t input_h = input_data->rows();
        const uint32_t input_w = input_data->cols();
        const uint32_t input_c = input_data->channels();

        // 默认 stride 和 pooling 大小一样
        const uint32_t pooling_h = uint32_t(std::floor(input_h / output_h_));
        const uint32_t pooling_w = uint32_t(std::floor(input_w / output_w_));

        std::cout << "pooling_h: " << pooling_h << ", pooling_w: " << pooling_w;

        std::shared_ptr<Tensor<float>> output_data = outputs.at(i);

        for (uint32_t ic = 0; ic < input_c; ++ic) {
            // 计算每个通道的pooling窗口平均值
            const arma::fmat& input_channel = input_data->slice(ic);
            arma::fmat& output_channel = output_data->slice(ic);

            // 数据是按列存储的，所以先遍历列，再遍历行，取地址更方便
            // w + pooling_w - 1 < input_w
            for (uint32_t c = 0; c + pooling_w - 1 < input_w; c += pooling_w) {
                for (uint32_t r = 0; r + pooling_h - 1 < input_h; r += pooling_h) {
                    float mean_value = 0.f;
                    // 该窗口的平均数
                    // 对应输出的列起始地址
                    float* output_channel_ptr = output_channel.colptr(int(c / pooling_w));
                    // 计算窗口的平均值
                    for (uint32_t w = 0; w < pooling_w; ++w) {
                        // 每次取出一个列的数据 - 第 c 个大窗口列里第 w 列，第 r 行大窗口
                        const float* col_ptr = input_channel.colptr(c + w) + r;
                        // 取出该窗口里 pooling_h 行的每一个数据
                        for (uint32_t h = 0; h < pooling_h; ++h) {
                            mean_value += *(col_ptr + h);
                        }
                    }
                    // output_channel_ptr 已经是列起始位置
                    mean_value /= float(pooling_h * pooling_w);
                    *(output_channel_ptr + r / pooling_h) = mean_value;
            }
        }
    }
}
}


void AdaptiveAvgPoolingLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op, std::shared_ptr<Layer>& adaptive_avgpooling_layer) {
    const auto& params = op->params;

    const auto& output_hw = dynamic_cast<RuntimeParameterIntArray*>(params.at("output_size"));

    if (!output_hw) {
        LOG(ERROR) << "output_size must be int array";
    }

    uint32_t output_h = output_hw->value.at(0);
    uint32_t output_w = output_hw->value.at(1);

    adaptive_avgpooling_layer = std::make_shared<kuiper_infer::AdaptiveAvgPoolingLayer>(output_h, output_w);
    
}


LayerRegisterWrapper kAdaptiveAvgPoolingLayer("nn.AdaptiveAvgPool2d", AdaptiveAvgPoolingLayer::CreateInstance);

}
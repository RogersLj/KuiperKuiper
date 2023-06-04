#include "./linear_layer.hpp"
#include <glog/logging.h>
#include "layer/layer_factory.hpp"
#include "data/tensor.hpp"

namespace kuiper_infer {
    
// 单一的特征向量一般为列向量 (in_features, 1)
// 因此权重矩阵的形状为(out_features, in_features)
// 计算时 - weight * input + bias - (out_features, 1)
// bias - (out_features, 1)
LinearLayer::LinearLayer(uint32_t in_features, uint32_t out_features, bool has_bias) : ParamLayer("LinearLayer"), in_features_(in_features), out_features_(out_features), has_bias_(has_bias) {
    this->InitWeightParam(1, 1, out_features, in_features);
    if (has_bias) {
        this->InitBiasParam(1, 1, 1, out_features);
        }
}

void LinearLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
    if (inputs.empty()) {
        LOG(FATAL) << "inputs is empty";
    }

    uint32_t batch_size = inputs.size();

    const std::shared_ptr<Tensor<float>> weights = this->weights_.at(0);
    // (1, out_features, in_features)

    arma::fmat weight_data(weights->data().memptr(), out_features_, in_features_);
    // 二​维矩阵
    // 转置
    const arma::fmat& weight_data_t = weight_data.t();

    for (uint32_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<float>> input = inputs.at(i);

        const std::vector<uint32_t> input_shape = input->shape();
        // (1, feature_dims, in_features)
        // 正常情况是 flatten 后的张量 - (1, 1, in_features)
        // feature_dims = 1

        const uint32_t feature_dims = input_shape.at(1);
        const uint32_t in_features = input_shape.at(2);
        CHECK(weight_data.n_rows == out_features_);
        CHECK(weight_data.n_cols == in_features && in_features == in_features_); 

        // 输入矩阵 - 例如 flatten 之后的形状
        // (1, 1, in_features)
        // 中间 shape.at(1) - feature_dims
        // 暂时不知道什么情况下不为 1
        arma::fmat input_vec((float*)input->raw_ptr(), feature_dims,in_features_);

        std::shared_ptr<Tensor<float>> output = outputs.at(i);

        if(output == nullptr || output->empty()) {
            output = std::make_shared<Tensor<float>>(1, out_features_, feature_dims);
            outputs.at(i) = output;
        }

        // feature_dims=1，所以通过raw_shape检查
        const auto& output_raw_shapes = output->raw_shape();
        if (output_raw_shapes.size() == 1) {
            CHECK(output_raw_shapes.at(0) == out_features_);
        }

        arma::fmat& result = output->slice(0);
        // 因为输入形状是 - (feature_dims, in_features)
        // weight - (out_features, in_features)
        // weight_t - (in_features, out_features)
        // output = input * weight_t - (feature_dims, out_features)
        result = input_vec * weight_data_t;

        if (has_bias_) {
            // 线性层的 bias 只有一个
            CHECK(!this->bias_.empty() && this->bias_.size() == 1) << "bias is empty but has_bias is true";
            // bias_data - (1, feature_dims, output_features)
            // 默认 - (1, 1, output_features)
            const auto& bias_data = this->bias_.front();

            CHECK(!bias_data->empty() && bias_data->channels() == 1 && bias_data->cols() == out_features_) << "The col of bias tensor is not same to output_features_";

            for (uint32_t i = 0; i < feature_dims; ++i) {
                result.row(i) += bias_data->slice(0);
            }

        }
        
    }
    
}

void LinearLayer::CreateInstance(const std::shared_ptr<RuntimeOperator> &op, std::shared_ptr<Layer>& linear_layer) {
    const auto& params = op->params;

    const auto& has_bias_param = dynamic_cast<RuntimeParameterBool*>(params.at("bias"));

    const auto& attrs = op->attrs;

    const auto& weight = attrs.at("weight");
    const auto& bias = attrs.at("bias");
    const auto& shape = weight->shape;

    int32_t out_features = shape.at(0);
    int32_t in_features = shape.at(1);
    const bool has_bias = has_bias_param->value;

    linear_layer = std::make_shared<LinearLayer>(in_features, out_features, has_bias);

    if (has_bias) {
        linear_layer->set_bias(bias->get<float>());
    }

    linear_layer->set_weights(weight->get<float>());
}

LayerRegisterWrapper kLinearCreateInstance("nn.Linear",LinearLayer::CreateInstance);

}
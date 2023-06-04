#include "./flatten.hpp"
#include "layer/layer_factory.hpp"
#include <glog/logging.h>
#include "data/tensor_util.hpp"


namespace kuiper_infer {
    
FlattenLayer::FlattenLayer(int start_dim, int end_dim) : Layer("FlattenLayer"), start_dim_(start_dim), end_dim_(end_dim) {
    
}


void FlattenLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>&inputs, std::vector<std::shared_ptr<Tensor<float>>> &outputs) {

    int start_dim = this->start_dim_;
    int end_dim = this->end_dim_;
    int total_dims = 4; // NCHW

    LOG(INFO) << "Flatten layer----------start_dim: " << start_dim << ", end_dim: " << end_dim;

    if (start_dim < 0) {
        start_dim = total_dims + start_dim;
    }

    if (end_dim < 0) {
        end_dim = total_dims + end_dim;
    }


    const uint32_t batch_size = inputs.size();

    for (uint32_t i = 0; i < batch_size; ++i) {
        const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);

        auto shape = input_data->shape();

        shape.insert(shape.begin(), batch_size);

        uint32_t elements_size =
        std::accumulate(shape.begin() + start_dim,
                        shape.begin() + end_dim + 1, 1, std::multiplies<int>());

        std::shared_ptr<Tensor<float>> output_data = outputs.at(i);

        output_data = TensorClone(input_data);

        outputs.at(i) = output_data;

        if (start_dim == 1 && end_dim == 3) {
            output_data->Reshape({elements_size}, true);
        } else if (start_dim == 2 && end_dim == 3) {
            uint32_t channels = shape.at(1);
            output_data->Reshape({channels, elements_size}, true);
        } else {
            LOG(FATAL) << "Unsupported flatten layer";
        }
    }

}


void FlattenLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,  std::shared_ptr<Layer>& flatten_layer) {

    const auto& params = op->params;

    const auto& start_dim = dynamic_cast<RuntimeParameterInt*>(params.at("start_dim"));
    const auto& end_dim = dynamic_cast<RuntimeParameterInt*>(params.at("end_dim"));

    flatten_layer = std::make_shared<FlattenLayer>(start_dim->value, end_dim->value);

}

LayerRegisterWrapper kFlattenLayer("torch.flatten", FlattenLayer::CreateInstance);


}
#include "parser_and_serializer/tflite/op_resolver.h"

#include "common/logging.h"
#include "parser_and_serializer/tflite/utils.h"

// This class serves for those operators who have no option.
class DummyOptionResolver : public BaseOptionResolver {
 public:
    virtual flatbuffers::Offset<void> SerializeOption(const Operator&                   op,
                                                      ::flatbuffers::FlatBufferBuilder* builder) const {
        DLOG(INFO) << "SKIP Export Option for " << ToStr(op.GetOpType());
        flatbuffers::Offset<void> dummy_option = 0;
        return dummy_option;
    }

    virtual void ParseOption(const void* builtin_options, Operator& op) const {
        DLOG(INFO) << "SKIP Import Option for " << ToStr(op.GetOpType());
    }
};

class Conv2DOptionResolver : public TfLiteOptionResolver<::tflite::Conv2DOptions, Conv2DOption> {
    void ParseOptionImpl(Conv2DOption& option, const ::tflite::Conv2DOptions& tflite_option) const final {
        option.stride_h        = tflite_option.stride_w();
        option.stride_w        = tflite_option.stride_w();
        option.dilation_w      = tflite_option.dilation_w_factor();
        option.dilation_h      = tflite_option.dilation_h_factor();
        option.pad_type        = utils::GetMappedPaddingOf(tflite_option.padding());
        option.activation_type = utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    flatbuffers::Offset<::tflite::Conv2DOptions>
    SerializeOptionImpl(const Conv2DOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto tflite_padding  = utils::GetMappedPaddingOf(option.pad_type);
        auto fuse_activation = utils::GetMappedActTypeOf(option.activation_type);
        return tflite::CreateConv2DOptions(*builder, tflite_padding, option.stride_w, option.stride_h, fuse_activation,
                                           option.dilation_w, option.dilation_h);
    }
};

class DepthwiseConv2DOptionResolver
    : public TfLiteOptionResolver<::tflite::DepthwiseConv2DOptions, DepthwiseConv2DOption> {
    void ParseOptionImpl(DepthwiseConv2DOption&                  option,
                         const ::tflite::DepthwiseConv2DOptions& tflite_option) const final {
        option.stride_h         = tflite_option.stride_w();
        option.stride_w         = tflite_option.stride_w();
        option.dilation_w       = tflite_option.dilation_w_factor();
        option.dilation_h       = tflite_option.dilation_h_factor();
        option.depth_multiplier = tflite_option.depth_multiplier();
        option.pad_type         = utils::GetMappedPaddingOf(tflite_option.padding());
        option.activation_type  = utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    flatbuffers::Offset<::tflite::DepthwiseConv2DOptions>
    SerializeOptionImpl(const DepthwiseConv2DOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto tflite_padding  = utils::GetMappedPaddingOf(option.pad_type);
        auto fuse_activation = utils::GetMappedActTypeOf(option.activation_type);
        return tflite::CreateDepthwiseConv2DOptions(*builder, tflite_padding, option.stride_w, option.stride_h,
                                                    option.depth_multiplier, fuse_activation, option.dilation_w,
                                                    option.dilation_h);
    }
};

class ReshapeOptionResolver : public TfLiteOptionResolver<::tflite::ReshapeOptions, ReshapeOption> {
    void ParseOptionImpl(ReshapeOption& option, const ::tflite::ReshapeOptions& tflite_option) const final {
        option.new_shape = utils::GetVecData<int>(tflite_option.new_shape());
    }

    flatbuffers::Offset<::tflite::ReshapeOptions>
    SerializeOptionImpl(const ReshapeOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateReshapeOptions(*builder, builder->CreateVector(option.new_shape));
    }
};

class SoftmaxOptionResolver : public TfLiteOptionResolver<::tflite::SoftmaxOptions, SoftmaxOption> {
    void ParseOptionImpl(SoftmaxOption& option, const ::tflite::SoftmaxOptions& tflite_option) const final {
        option.beta = tflite_option.beta();
    }

    flatbuffers::Offset<::tflite::SoftmaxOptions>
    SerializeOptionImpl(const SoftmaxOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateSoftmaxOptions(*builder, option.beta);
    }
};

class Pool2DOptionOptionResolver : public TfLiteOptionResolver<::tflite::Pool2DOptions, Pool2DOption> {
    void ParseOptionImpl(Pool2DOption& option, const ::tflite::Pool2DOptions& tflite_option) const final {
        option.stride_h = tflite_option.stride_h();
        option.stride_w = tflite_option.stride_w();
        option.filter_h = tflite_option.filter_height();
        option.filter_w = tflite_option.filter_width();
        option.pad_type = utils::GetMappedPaddingOf(tflite_option.padding());
    }

    flatbuffers::Offset<::tflite::Pool2DOptions>
    SerializeOptionImpl(const Pool2DOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto tflite_padding = utils::GetMappedPaddingOf(option.pad_type);
        return tflite::CreatePool2DOptions(*builder, tflite_padding, option.stride_w, option.stride_h, option.filter_w,
                                           option.filter_h);
    }
};

OperatorResolver::OperatorResolver() {
    AddOpResolver<Conv2DOptionResolver>(OperatorType::CONV2D, ::tflite::BuiltinOperator_CONV_2D,
                                        ::tflite::BuiltinOptions_Conv2DOptions);
    AddOpResolver<DepthwiseConv2DOptionResolver>(OperatorType::DEPTHWISE_CONV2D,
                                                 ::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                                                 ::tflite::BuiltinOptions_DepthwiseConv2DOptions);
    AddOpResolver<Pool2DOptionOptionResolver>(OperatorType::MAX_POOL, ::tflite::BuiltinOperator_MAX_POOL_2D,
                                              ::tflite::BuiltinOptions_Pool2DOptions);
    AddOpResolver<Pool2DOptionOptionResolver>(OperatorType::AVERAGE_POOL, ::tflite::BuiltinOperator_AVERAGE_POOL_2D,
                                              ::tflite::BuiltinOptions_Pool2DOptions);
    AddOpResolver<SoftmaxOptionResolver>(OperatorType::SOFTMAX, ::tflite::BuiltinOperator_SOFTMAX,
                                         ::tflite::BuiltinOptions_SoftmaxOptions);
    AddOpResolver<ReshapeOptionResolver>(OperatorType::RESHAPE, ::tflite::BuiltinOperator_RESHAPE,
                                         ::tflite::BuiltinOptions_ReshapeOptions);
}

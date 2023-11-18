#include "ir_port/tflite_op_resolver.h"

#include "common/logging.h"
#include "ir_port/parser_utils.h"

// This class serves for those operators who have no option.
class DummyOptionResolver : public BaseOptionResolver {
 public:
    virtual void SerializeOption(const Operator&                   op,
                                 ::flatbuffers::FlatBufferBuilder* builder) const {
        DLOG(INFO) << "SKIP Export Option for " << ToStr(op.GetOpType());
    }

    virtual void ParseOption(const void* builtin_options, Operator& op) const {
        DLOG(INFO) << "SKIP Import Option for " << ToStr(op.GetOpType());
    }
};

class Conv2DOptionResolver : public TfLiteOptionResolver<::tflite::Conv2DOptions, Conv2DOption> {
    void ParseOptionImpl(Conv2DOption&                  option,
                         const ::tflite::Conv2DOptions& tflite_option) const final {
        option.stride_h   = tflite_option.stride_w();
        option.stride_w   = tflite_option.stride_w();
        option.dilation_w = tflite_option.dilation_w_factor();
        option.dilation_h = tflite_option.dilation_h_factor();
        option.pad_type   = utils::GetMappedPaddingOf(tflite_option.padding());
        option.activation_type =
            utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    void SerializeOptionImpl(const Conv2DOption&,
                             ::flatbuffers::FlatBufferBuilder* builder) const final {}
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
        option.activation_type =
            utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    void SerializeOptionImpl(const DepthwiseConv2DOption&,
                             ::flatbuffers::FlatBufferBuilder* builder) const final {}
};

class ReshapeOptionResolver : public TfLiteOptionResolver<::tflite::ReshapeOptions, ReshapeOption> {
    void ParseOptionImpl(ReshapeOption&                  option,
                         const ::tflite::ReshapeOptions& tflite_option) const final {
        option.new_shape = utils::GetVecData<int>(tflite_option.new_shape());
    }

    void SerializeOptionImpl(const ReshapeOption&,
                             ::flatbuffers::FlatBufferBuilder* builder) const final {}
};

class SoftmaxOptionResolver : public TfLiteOptionResolver<::tflite::SoftmaxOptions, SoftmaxOption> {
    void ParseOptionImpl(SoftmaxOption&                  option,
                         const ::tflite::SoftmaxOptions& tflite_option) const final {
        option.beta = tflite_option.beta();
    }

    void SerializeOptionImpl(const SoftmaxOption&,
                             ::flatbuffers::FlatBufferBuilder* builder) const final {}
};

class Pool2DOptionOptionResolver
    : public TfLiteOptionResolver<::tflite::Pool2DOptions, Pool2DOption> {
    void ParseOptionImpl(Pool2DOption&                  option,
                         const ::tflite::Pool2DOptions& tflite_option) const final {
        option.stride_h = tflite_option.stride_h();
        option.stride_w = tflite_option.stride_w();
        option.filter_h = tflite_option.filter_height();
        option.filter_w = tflite_option.filter_width();
        option.pad_type = utils::GetMappedPaddingOf(tflite_option.padding());
    }

    void SerializeOptionImpl(const Pool2DOption&,
                             ::flatbuffers::FlatBufferBuilder* builder) const final {}
};

OperatorResolver::OperatorResolver() {
    AddOpResolver<Conv2DOptionResolver>(OperatorType::CONV2D, ::tflite::BuiltinOperator_CONV_2D);
    AddOpResolver<DepthwiseConv2DOptionResolver>(OperatorType::DEPTHWISE_CONV2D,
                                                 ::tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
    AddOpResolver<Pool2DOptionOptionResolver>(OperatorType::MAX_POOL,
                                              ::tflite::BuiltinOperator_MAX_POOL_2D);
    AddOpResolver<Pool2DOptionOptionResolver>(OperatorType::AVERAGE_POOL,
                                              ::tflite::BuiltinOperator_AVERAGE_POOL_2D);
    AddOpResolver<SoftmaxOptionResolver>(OperatorType::SOFTMAX, ::tflite::BuiltinOperator_SOFTMAX);
    AddOpResolver<ReshapeOptionResolver>(OperatorType::RESHAPE, ::tflite::BuiltinOperator_RESHAPE);
}

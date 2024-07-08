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

class TransposeConv2DOptionResolver
    : public TfLiteOptionResolver<::tflite::TransposeConvOptions, TransposeConv2DOption> {
    void ParseOptionImpl(TransposeConv2DOption&                option,
                         const ::tflite::TransposeConvOptions& tflite_option) const final {
        option.stride_h        = tflite_option.stride_w();
        option.stride_w        = tflite_option.stride_w();
        option.pad_type        = utils::GetMappedPaddingOf(tflite_option.padding());
        option.activation_type = utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    flatbuffers::Offset<::tflite::TransposeConvOptions>
    SerializeOptionImpl(const TransposeConv2DOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto tflite_padding  = utils::GetMappedPaddingOf(option.pad_type);
        auto fuse_activation = utils::GetMappedActTypeOf(option.activation_type);
        return tflite::CreateTransposeConvOptions(*builder, tflite_padding, option.stride_w, option.stride_h,
                                                  fuse_activation);
    }
};

class Conv3DOptionResolver : public TfLiteOptionResolver<::tflite::Conv3DOptions, Conv3DOption> {
    void ParseOptionImpl(Conv3DOption& option, const ::tflite::Conv3DOptions& tflite_option) const final {
        option.stride_d          = tflite_option.stride_d();
        option.stride_w          = tflite_option.stride_w();
        option.stride_h          = tflite_option.stride_h();
        option.dilation_d_factor = tflite_option.dilation_d_factor();
        option.dilation_w_factor = tflite_option.dilation_w_factor();
        option.dilation_h_factor = tflite_option.dilation_h_factor();
        option.pad_type          = utils::GetMappedPaddingOf(tflite_option.padding());
        option.activation_type   = utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    flatbuffers::Offset<::tflite::Conv3DOptions>
    SerializeOptionImpl(const Conv3DOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto tflite_padding  = utils::GetMappedPaddingOf(option.pad_type);
        auto fuse_activation = utils::GetMappedActTypeOf(option.activation_type);
        return tflite::CreateConv3DOptions(*builder, tflite_padding, option.stride_d, option.stride_w, option.stride_h,
                                           fuse_activation, option.dilation_d_factor, option.dilation_w_factor,
                                           option.dilation_h_factor);
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

class AddOptionResolver : public TfLiteOptionResolver<::tflite::AddOptions, AddOption> {
    void ParseOptionImpl(AddOption& option, const ::tflite::AddOptions& tflite_option) const final {
        option.pot_scale_int16 = tflite_option.pot_scale_int16();
        option.activation_type = utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    flatbuffers::Offset<::tflite::AddOptions>
    SerializeOptionImpl(const AddOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto fuse_activation = utils::GetMappedActTypeOf(option.activation_type);
        return tflite::CreateAddOptions(*builder, fuse_activation, option.pot_scale_int16);
    }
};

class MulOptionResolver : public TfLiteOptionResolver<::tflite::MulOptions, MulOption> {
    void ParseOptionImpl(MulOption& option, const ::tflite::MulOptions& tflite_option) const final {
        option.activation_type = utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    flatbuffers::Offset<::tflite::MulOptions>
    SerializeOptionImpl(const MulOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateMulOptions(*builder, utils::GetMappedActTypeOf(option.activation_type));
    }
};

class SubOptionResolver : public TfLiteOptionResolver<::tflite::SubOptions, SubOption> {
    void ParseOptionImpl(SubOption& option, const ::tflite::SubOptions& tflite_option) const final {
        option.pot_scale_int16 = tflite_option.pot_scale_int16();
        option.activation_type = utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    flatbuffers::Offset<::tflite::SubOptions>
    SerializeOptionImpl(const SubOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto fuse_activation = utils::GetMappedActTypeOf(option.activation_type);
        return tflite::CreateSubOptions(*builder, fuse_activation, option.pot_scale_int16);
    }
};

class DivOptionResolver : public TfLiteOptionResolver<::tflite::DivOptions, DivOption> {
    void ParseOptionImpl(DivOption& option, const ::tflite::DivOptions& tflite_option) const final {
        option.activation_type = utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    flatbuffers::Offset<::tflite::DivOptions>
    SerializeOptionImpl(const DivOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateDivOptions(*builder, utils::GetMappedActTypeOf(option.activation_type));
    }
};

class ConcatOptionResolver : public TfLiteOptionResolver<::tflite::ConcatenationOptions, ConcatOption> {
    void ParseOptionImpl(ConcatOption& option, const ::tflite::ConcatenationOptions& tflite_option) const final {
        option.axis            = tflite_option.axis();
        option.activation_type = utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    flatbuffers::Offset<::tflite::ConcatenationOptions>
    SerializeOptionImpl(const ConcatOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto fuse_activation = utils::GetMappedActTypeOf(option.activation_type);
        return tflite::CreateConcatenationOptions(*builder, option.axis, fuse_activation);
    }
};

class ResizeBilinearOptionResolver : public TfLiteOptionResolver<::tflite::ResizeBilinearOptions, ResizeOption> {
    void ParseOptionImpl(ResizeOption& option, const ::tflite::ResizeBilinearOptions& tflite_option) const final {
        option.align_corners      = tflite_option.align_corners();
        option.half_pixel_centers = tflite_option.half_pixel_centers();
    }

    flatbuffers::Offset<::tflite::ResizeBilinearOptions>
    SerializeOptionImpl(const ResizeOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateResizeBilinearOptions(*builder, option.align_corners, option.half_pixel_centers);
    }
};

class ResizeNeighborOptionResolver : public TfLiteOptionResolver<::tflite::ResizeNearestNeighborOptions, ResizeOption> {
    void ParseOptionImpl(ResizeOption&                                 option,
                         const ::tflite::ResizeNearestNeighborOptions& tflite_option) const final {
        option.align_corners      = tflite_option.align_corners();
        option.half_pixel_centers = tflite_option.half_pixel_centers();
    }

    flatbuffers::Offset<::tflite::ResizeNearestNeighborOptions>
    SerializeOptionImpl(const ResizeOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateResizeNearestNeighborOptions(*builder, option.align_corners, option.half_pixel_centers);
    }
};

class FullyConnectedOptionResolver
    : public TfLiteOptionResolver<::tflite::FullyConnectedOptions, FullyConnectedOption> {
    void ParseOptionImpl(FullyConnectedOption&                  option,
                         const ::tflite::FullyConnectedOptions& tflite_option) const final {
        option.keep_num_dims        = tflite_option.keep_num_dims();
        option.asym_quantize_inputs = tflite_option.asymmetric_quantize_inputs();
        option.activation_type      = utils::GetMappedActTypeOf(tflite_option.fused_activation_function());
    }

    flatbuffers::Offset<::tflite::FullyConnectedOptions>
    SerializeOptionImpl(const FullyConnectedOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto fuse_activation = utils::GetMappedActTypeOf(option.activation_type);
        return tflite::CreateFullyConnectedOptions(*builder, fuse_activation,
                                                   tflite::FullyConnectedOptionsWeightsFormat_DEFAULT,
                                                   option.keep_num_dims, option.asym_quantize_inputs);
    }
};

class PackOptionResolver : public TfLiteOptionResolver<::tflite::PackOptions, PackOption> {
    void ParseOptionImpl(PackOption& option, const ::tflite::PackOptions& tflite_option) const final {
        option.axis         = tflite_option.axis();
        option.values_count = tflite_option.values_count();
    }

    flatbuffers::Offset<::tflite::PackOptions>
    SerializeOptionImpl(const PackOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreatePackOptions(*builder, option.values_count, option.axis);
    }
};

class UnpackOptionResolver : public TfLiteOptionResolver<::tflite::UnpackOptions, UnPackOption> {
    void ParseOptionImpl(UnPackOption& option, const ::tflite::UnpackOptions& tflite_option) const final {
        option.axis   = tflite_option.axis();
        option.number = tflite_option.num();
    }

    flatbuffers::Offset<::tflite::UnpackOptions>
    SerializeOptionImpl(const UnPackOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateUnpackOptions(*builder, option.number, option.axis);
    }
};

class CastOptionResolver : public TfLiteOptionResolver<::tflite::CastOptions, CastOption> {
    void ParseOptionImpl(CastOption& option, const ::tflite::CastOptions& tflite_option) const final {
        option.in_data_type  = utils::GetMappedDataTypeOf(tflite_option.in_data_type());
        option.out_data_type = utils::GetMappedDataTypeOf(tflite_option.out_data_type());
    }

    flatbuffers::Offset<::tflite::CastOptions>
    SerializeOptionImpl(const CastOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto in_data_type  = utils::GetMappedDataTypeOf(option.in_data_type);
        auto out_data_type = utils::GetMappedDataTypeOf(option.out_data_type);
        return tflite::CreateCastOptions(*builder, in_data_type, out_data_type);
    }
};

class GatherOptionResolver : public TfLiteOptionResolver<::tflite::GatherOptions, GatherOption> {
    void ParseOptionImpl(GatherOption& option, const ::tflite::GatherOptions& tflite_option) const final {
        option.axis       = tflite_option.axis();
        option.batch_dims = tflite_option.batch_dims();
    }

    flatbuffers::Offset<::tflite::GatherOptions>
    SerializeOptionImpl(const GatherOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateGatherOptions(*builder, option.axis, option.batch_dims);
    }
};

class DepthToSpaceOptionResolver : public TfLiteOptionResolver<::tflite::DepthToSpaceOptions, DepthToSpaceOption> {
    void ParseOptionImpl(DepthToSpaceOption& option, const ::tflite::DepthToSpaceOptions& tflite_option) const final {
        option.block_size = tflite_option.block_size();
    }

    flatbuffers::Offset<::tflite::DepthToSpaceOptions>
    SerializeOptionImpl(const DepthToSpaceOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateDepthToSpaceOptions(*builder, option.block_size);
    }
};

class SpaceToDepthOptionResolver : public TfLiteOptionResolver<::tflite::SpaceToDepthOptions, SpaceToDepthOption> {
    void ParseOptionImpl(SpaceToDepthOption& option, const ::tflite::SpaceToDepthOptions& tflite_option) const final {
        option.block_size = tflite_option.block_size();
    }

    flatbuffers::Offset<::tflite::SpaceToDepthOptions>
    SerializeOptionImpl(const SpaceToDepthOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateSpaceToDepthOptions(*builder, option.block_size);
    }
};

class StridedSliceOptionResolver : public TfLiteOptionResolver<::tflite::StridedSliceOptions, StridedSliceOption> {
    void ParseOptionImpl(StridedSliceOption& option, const ::tflite::StridedSliceOptions& tflite_option) const final {
        option.begin_mask       = tflite_option.begin_mask();
        option.end_mask         = tflite_option.end_mask();
        option.ellipsis_mask    = tflite_option.ellipsis_mask();
        option.new_axis_mask    = tflite_option.new_axis_mask();
        option.shrink_axis_mask = tflite_option.shrink_axis_mask();
        option.offset           = tflite_option.offset();
    }

    flatbuffers::Offset<::tflite::StridedSliceOptions>
    SerializeOptionImpl(const StridedSliceOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateStridedSliceOptions(*builder, option.begin_mask, option.end_mask, option.ellipsis_mask,
                                                 option.new_axis_mask, option.shrink_axis_mask, option.offset);
    }
};

class SplitOptionResolver : public TfLiteOptionResolver<::tflite::SplitOptions, SplitOption> {
    void ParseOptionImpl(SplitOption& option, const ::tflite::SplitOptions& tflite_option) const final {
        option.num_splits = tflite_option.num_splits();
    }

    flatbuffers::Offset<::tflite::SplitOptions>
    SerializeOptionImpl(const SplitOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateSplitOptions(*builder, option.num_splits);
    }
};

class SplitVOptionResolver : public TfLiteOptionResolver<::tflite::SplitVOptions, SplitOption> {
    void ParseOptionImpl(SplitOption& option, const ::tflite::SplitVOptions& tflite_option) const final {
        option.num_splits = tflite_option.num_splits();
    }

    flatbuffers::Offset<::tflite::SplitVOptions>
    SerializeOptionImpl(const SplitOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateSplitVOptions(*builder, option.num_splits);
    }
};

class LeakyReLUOptionResolver : public TfLiteOptionResolver<::tflite::LeakyReluOptions, LeakyReLUOption> {
    void ParseOptionImpl(LeakyReLUOption& option, const ::tflite::LeakyReluOptions& tflite_option) const final {
        option.alpha = tflite_option.alpha();
    }

    flatbuffers::Offset<::tflite::LeakyReluOptions>
    SerializeOptionImpl(const LeakyReLUOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateLeakyReluOptions(*builder, option.alpha);
    }
};

class MirrorPadOptionResolver : public TfLiteOptionResolver<::tflite::MirrorPadOptions, MirrorPadOption> {
    void ParseOptionImpl(MirrorPadOption& option, const ::tflite::MirrorPadOptions& tflite_option) const final {
        option.mirror_pad_type = static_cast<MirrorPadType>(tflite_option.mode());
    }

    flatbuffers::Offset<::tflite::MirrorPadOptions>
    SerializeOptionImpl(const MirrorPadOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        auto mirror_pad_type = static_cast<tflite::MirrorPadMode>(option.mirror_pad_type);
        return tflite::CreateMirrorPadOptions(*builder, mirror_pad_type);
    }
};

class GELUOptionResolver : public TfLiteOptionResolver<::tflite::GeluOptions, GELUOption> {
    void ParseOptionImpl(GELUOption& option, const ::tflite::GeluOptions& tflite_option) const final {
        option.approximate = tflite_option.approximate();
    }

    flatbuffers::Offset<::tflite::GeluOptions>
    SerializeOptionImpl(const GELUOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateGeluOptions(*builder, option.approximate);
    }
};

class OneHotOptionResolver : public TfLiteOptionResolver<::tflite::OneHotOptions, OneHotOption> {
    void ParseOptionImpl(OneHotOption& option, const ::tflite::OneHotOptions& tflite_option) const final {
        option.axis = tflite_option.axis();
    }

    flatbuffers::Offset<::tflite::OneHotOptions>
    SerializeOptionImpl(const OneHotOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateOneHotOptions(*builder, option.axis);
    }
};

class ShapeOptionResolver : public TfLiteOptionResolver<::tflite::ShapeOptions, ShapeOption> {
    void ParseOptionImpl(ShapeOption& option, const ::tflite::ShapeOptions& tflite_option) const final {
        option.out_type = utils::GetMappedDataTypeOf(tflite_option.out_type());
    }

    flatbuffers::Offset<::tflite::ShapeOptions>
    SerializeOptionImpl(const ShapeOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateShapeOptions(*builder, utils::GetMappedDataTypeOf(option.out_type));
    }
};

class ArgMinOptionResolver : public TfLiteOptionResolver<::tflite::ArgMinOptions, ArgMinOption> {
    void ParseOptionImpl(ArgMinOption& option, const ::tflite::ArgMinOptions& tflite_option) const final {
        option.out_type = utils::GetMappedDataTypeOf(tflite_option.output_type());
    }

    flatbuffers::Offset<::tflite::ArgMinOptions>
    SerializeOptionImpl(const ArgMinOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateArgMinOptions(*builder, utils::GetMappedDataTypeOf(option.out_type));
    }
};

class ArgMaxOptionResolver : public TfLiteOptionResolver<::tflite::ArgMaxOptions, ArgMaxOption> {
    void ParseOptionImpl(ArgMaxOption& option, const ::tflite::ArgMaxOptions& tflite_option) const final {
        option.out_type = utils::GetMappedDataTypeOf(tflite_option.output_type());
    }

    flatbuffers::Offset<::tflite::ArgMaxOptions>
    SerializeOptionImpl(const ArgMaxOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateArgMaxOptions(*builder, utils::GetMappedDataTypeOf(option.out_type));
    }
};

class BatchMatmulOptionResolver : public TfLiteOptionResolver<::tflite::BatchMatMulOptions, BatchMatmulOption> {
    void ParseOptionImpl(BatchMatmulOption& option, const ::tflite::BatchMatMulOptions& tflite_option) const final {
        option.adj_x                = tflite_option.adj_x();
        option.adj_y                = tflite_option.adj_y();
        option.asym_quantize_inputs = tflite_option.asymmetric_quantize_inputs();
    }

    flatbuffers::Offset<::tflite::BatchMatMulOptions>
    SerializeOptionImpl(const BatchMatmulOption& option, ::flatbuffers::FlatBufferBuilder* builder) const final {
        return tflite::CreateBatchMatMulOptions(*builder, option.adj_x, option.adj_y, option.asym_quantize_inputs);
    }
};

/**
 * OperatorResolver provides option parsers for each type of operator.
 * Create each parser in OperatorResolver's constructor, please add new type below.
 */
OperatorResolver::OperatorResolver() {
    AddOpResolver<Conv2DOptionResolver>(OperatorType::CONV2D, ::tflite::BuiltinOperator_CONV_2D,
                                        ::tflite::BuiltinOptions_Conv2DOptions);
    AddOpResolver<DepthwiseConv2DOptionResolver>(OperatorType::DEPTHWISE_CONV2D,
                                                 ::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                                                 ::tflite::BuiltinOptions_DepthwiseConv2DOptions);
    AddOpResolver<TransposeConv2DOptionResolver>(OperatorType::TRANSPOSE_CONV2D,
                                                 ::tflite::BuiltinOperator_TRANSPOSE_CONV,
                                                 ::tflite::BuiltinOptions_TransposeConvOptions);
    AddOpResolver<Conv3DOptionResolver>(OperatorType::CONV3D, ::tflite::BuiltinOperator_CONV_3D,
                                        ::tflite::BuiltinOptions_Conv3DOptions);
    AddOpResolver<Pool2DOptionOptionResolver>(OperatorType::MAX_POOL, ::tflite::BuiltinOperator_MAX_POOL_2D,
                                              ::tflite::BuiltinOptions_Pool2DOptions);
    AddOpResolver<Pool2DOptionOptionResolver>(OperatorType::AVERAGE_POOL, ::tflite::BuiltinOperator_AVERAGE_POOL_2D,
                                              ::tflite::BuiltinOptions_Pool2DOptions);
    AddOpResolver<SoftmaxOptionResolver>(OperatorType::SOFTMAX, ::tflite::BuiltinOperator_SOFTMAX,
                                         ::tflite::BuiltinOptions_SoftmaxOptions);
    AddOpResolver<ReshapeOptionResolver>(OperatorType::RESHAPE, ::tflite::BuiltinOperator_RESHAPE,
                                         ::tflite::BuiltinOptions_ReshapeOptions);
    AddOpResolver<AddOptionResolver>(OperatorType::ADD, ::tflite::BuiltinOperator_ADD,
                                     ::tflite::BuiltinOptions_AddOptions);
    AddOpResolver<MulOptionResolver>(OperatorType::MUL, ::tflite::BuiltinOperator_MUL,
                                     ::tflite::BuiltinOptions_MulOptions);
    AddOpResolver<SubOptionResolver>(OperatorType::SUB, ::tflite::BuiltinOperator_SUB,
                                     ::tflite::BuiltinOptions_SubOptions);
    AddOpResolver<DivOptionResolver>(OperatorType::DIV, ::tflite::BuiltinOperator_DIV,
                                     ::tflite::BuiltinOptions_DivOptions);
    AddOpResolver<ConcatOptionResolver>(OperatorType::CONCAT, ::tflite::BuiltinOperator_CONCATENATION,
                                        ::tflite::BuiltinOptions_ConcatenationOptions);
    AddOpResolver<ResizeBilinearOptionResolver>(OperatorType::RESIZE_BILINEAR,
                                                ::tflite::BuiltinOperator_RESIZE_BILINEAR,
                                                ::tflite::BuiltinOptions_ResizeBilinearOptions);
    AddOpResolver<ResizeNeighborOptionResolver>(OperatorType::RESIZE_NEAREST_NEIGHBOR,
                                                ::tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                                                ::tflite::BuiltinOptions_ResizeNearestNeighborOptions);
    AddOpResolver<FullyConnectedOptionResolver>(OperatorType::FULLY_CONNECTED,
                                                ::tflite::BuiltinOperator_FULLY_CONNECTED,
                                                ::tflite::BuiltinOptions_FullyConnectedOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::EXP, ::tflite::BuiltinOperator_EXP,
                                       ::tflite::BuiltinOptions_ExpOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::MEAN, ::tflite::BuiltinOperator_MEAN,
                                       ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::TRANSPOSE, ::tflite::BuiltinOperator_TRANSPOSE,
                                       ::tflite::BuiltinOptions_TransposeOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::DEQUANTIZE, ::tflite::BuiltinOperator_DEQUANTIZE,
                                       ::tflite::BuiltinOptions_DequantizeOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::QUANTIZE, ::tflite::BuiltinOperator_QUANTIZE,
                                       ::tflite::BuiltinOptions_QuantizeOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::SLICE, ::tflite::BuiltinOperator_SLICE,
                                       ::tflite::BuiltinOptions_SliceOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::TILE, ::tflite::BuiltinOperator_TILE,
                                       ::tflite::BuiltinOptions_TileOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::POW, ::tflite::BuiltinOperator_POW,
                                       ::tflite::BuiltinOptions_PowOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::ABS, ::tflite::BuiltinOperator_ABS,
                                       ::tflite::BuiltinOptions_AbsOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::HARDSWISH, ::tflite::BuiltinOperator_HARD_SWISH,
                                       ::tflite::BuiltinOptions_HardSwishOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::TANH, ::tflite::BuiltinOperator_TANH,
                                       ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::SQUARE, ::tflite::BuiltinOperator_SQUARE,
                                       ::tflite::BuiltinOptions_SquareOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::SQUARED_DIFFERENCE, ::tflite::BuiltinOperator_SQUARED_DIFFERENCE,
                                       ::tflite::BuiltinOptions_SquaredDifferenceOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::BATCH_TO_SPACE_ND, ::tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
                                       ::tflite::BuiltinOptions_BatchToSpaceNDOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::SPACE_TO_BATCH_ND, ::tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
                                       ::tflite::BuiltinOptions_SpaceToBatchNDOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::TOPK_V2, ::tflite::BuiltinOperator_TOPK_V2,
                                       ::tflite::BuiltinOptions_TopKV2Options);
    AddOpResolver<DummyOptionResolver>(OperatorType::LOG, ::tflite::BuiltinOperator_LOG, ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::LOG_SOFTMAX, ::tflite::BuiltinOperator_LOG_SOFTMAX,
                                       ::tflite::BuiltinOptions_LogSoftmaxOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::NEG, ::tflite::BuiltinOperator_NEG,
                                       ::tflite::BuiltinOptions_NegOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::SUM, ::tflite::BuiltinOperator_SUM, ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::SIN, ::tflite::BuiltinOperator_SIN, ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::COSINE, ::tflite::BuiltinOperator_COS,
                                       ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::PADV2, ::tflite::BuiltinOperator_PADV2,
                                       ::tflite::BuiltinOptions_PadV2Options);

    AddOpResolver<PackOptionResolver>(OperatorType::PACK, ::tflite::BuiltinOperator_PACK,
                                      ::tflite::BuiltinOptions_PackOptions);
    AddOpResolver<UnpackOptionResolver>(OperatorType::UNPACK, ::tflite::BuiltinOperator_UNPACK,
                                        ::tflite::BuiltinOptions_UnpackOptions);
    AddOpResolver<CastOptionResolver>(OperatorType::CAST, ::tflite::BuiltinOperator_CAST,
                                      ::tflite::BuiltinOptions_CastOptions);
    AddOpResolver<GatherOptionResolver>(OperatorType::GATHER, ::tflite::BuiltinOperator_GATHER,
                                        ::tflite::BuiltinOptions_GatherOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::LOGISTIC, ::tflite::BuiltinOperator_LOGISTIC,
                                       ::tflite::BuiltinOptions_NONE);
    AddOpResolver<SplitOptionResolver>(OperatorType::SPLIT, ::tflite::BuiltinOperator_SPLIT,
                                       ::tflite::BuiltinOptions_SplitOptions);
    AddOpResolver<SplitVOptionResolver>(OperatorType::SPLITV, ::tflite::BuiltinOperator_SPLIT_V,
                                        ::tflite::BuiltinOptions_SplitVOptions);
    AddOpResolver<StridedSliceOptionResolver>(OperatorType::STRIDED_SLICE, ::tflite::BuiltinOperator_STRIDED_SLICE,
                                              ::tflite::BuiltinOptions_StridedSliceOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::SQRT, ::tflite::BuiltinOperator_SQRT,
                                       ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::ELU, ::tflite::BuiltinOperator_ELU, ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::ReLU, ::tflite::BuiltinOperator_RELU,
                                       ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::ReLU6, ::tflite::BuiltinOperator_RELU6,
                                       ::tflite::BuiltinOptions_NONE);
    AddOpResolver<LeakyReLUOptionResolver>(OperatorType::LEAKY_RELU, ::tflite::BuiltinOperator_LEAKY_RELU,
                                           ::tflite::BuiltinOptions_LeakyReluOptions);
    AddOpResolver<MirrorPadOptionResolver>(OperatorType::MIRROR_PAD, ::tflite::BuiltinOperator_MIRROR_PAD,
                                           ::tflite::BuiltinOptions_MirrorPadOptions);
    AddOpResolver<GELUOptionResolver>(OperatorType::GELU, ::tflite::BuiltinOperator_GELU,
                                      ::tflite::BuiltinOptions_GeluOptions);
    AddOpResolver<OneHotOptionResolver>(OperatorType::ONE_HOT, ::tflite::BuiltinOperator_ONE_HOT,
                                        ::tflite::BuiltinOptions_OneHotOptions);
    AddOpResolver<SpaceToDepthOptionResolver>(OperatorType::SPACE_TO_DEPTH, ::tflite::BuiltinOperator_SPACE_TO_DEPTH,
                                              ::tflite::BuiltinOptions_SpaceToDepthOptions);
    AddOpResolver<DepthToSpaceOptionResolver>(OperatorType::DEPTH_TO_SPACE, ::tflite::BuiltinOperator_DEPTH_TO_SPACE,
                                              ::tflite::BuiltinOptions_DepthToSpaceOptions);

    AddOpResolver<BatchMatmulOptionResolver>(OperatorType::BATCH_MATMUL, ::tflite::BuiltinOperator_BATCH_MATMUL,
                                             ::tflite::BuiltinOptions_BatchMatMulOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::PReLU, ::tflite::BuiltinOperator_PRELU,
                                       ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::PAD, ::tflite::BuiltinOperator_PAD, ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::RSQRT, ::tflite::BuiltinOperator_RSQRT,
                                       ::tflite::BuiltinOptions_NONE);
    AddOpResolver<DummyOptionResolver>(OperatorType::MAXIMUM, ::tflite::BuiltinOperator_MAXIMUM,
                                       ::tflite::BuiltinOptions_MaximumMinimumOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::MINIMUM, ::tflite::BuiltinOperator_MINIMUM,
                                       ::tflite::BuiltinOptions_MaximumMinimumOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::ARGMIN, ::tflite::BuiltinOperator_ARG_MIN,
                                       ::tflite::BuiltinOptions_ArgMinOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::ARGMAX, ::tflite::BuiltinOperator_ARG_MAX,
                                       ::tflite::BuiltinOptions_ArgMaxOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::BROADCAST_TO, ::tflite::BuiltinOperator_BROADCAST_TO,
                                       ::tflite::BuiltinOptions_BroadcastToOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::GATHER_ND, ::tflite::BuiltinOperator_GATHER_ND,
                                       ::tflite::BuiltinOptions_GatherNdOptions);
    AddOpResolver<DummyOptionResolver>(OperatorType::SELECT_V2, ::tflite::BuiltinOperator_SELECT_V2,
                                       ::tflite::BuiltinOptions_SelectV2Options);
}

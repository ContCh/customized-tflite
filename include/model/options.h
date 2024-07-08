#pragma once

#include <vector>

#include "model/types.h"

struct BaseOption {};

/// Define necessary param type in option
enum Padding { SAME = 0, VALID = 1 };

enum MirrorPadType { REFLECT = 0, SYMMETRIC = 1 };

/// Options based on op type
struct Conv2DOption : public BaseOption {
    int          stride_w;
    int          stride_h;
    int          dilation_w;
    int          dilation_h;
    Padding      pad_type;
    OperatorType activation_type;
};

struct Pool2DOption : public BaseOption {
    int     stride_w;
    int     stride_h;
    int     filter_w;
    int     filter_h;
    Padding pad_type;
};

struct DepthwiseConv2DOption : public BaseOption {
    int          stride_w;
    int          stride_h;
    int          dilation_w;
    int          dilation_h;
    int          depth_multiplier;
    Padding      pad_type;
    OperatorType activation_type;
};

struct TransposeConv2DOption : public BaseOption {
    int          stride_w;
    int          stride_h;
    Padding      pad_type;
    OperatorType activation_type;
};

struct Conv3DOption : public BaseOption {
    int          stride_d;
    int          stride_w;
    int          stride_h;
    int          dilation_d_factor;
    int          dilation_w_factor;
    int          dilation_h_factor;
    Padding      pad_type;
    OperatorType activation_type;
};

struct ReshapeOption : public BaseOption {
    std::vector<int32_t> new_shape;
};

struct SoftmaxOption : public BaseOption {
    float beta;
};

struct AddOption : public BaseOption {
    bool         pot_scale_int16;
    OperatorType activation_type;
};

struct MulOption : public BaseOption {
    OperatorType activation_type;
};

struct SubOption : public BaseOption {
    bool         pot_scale_int16;
    OperatorType activation_type;
};

struct DivOption : public BaseOption {
    OperatorType activation_type;
};

struct ConcatOption : public BaseOption {
    int          axis;
    OperatorType activation_type;
};

struct ResizeOption : public BaseOption {
    bool align_corners;
    bool half_pixel_centers;
};

struct FullyConnectedOption : public BaseOption {
    bool         keep_num_dims;
    bool         asym_quantize_inputs;
    OperatorType activation_type;
};

struct PackOption : public BaseOption {
    int values_count;
    int axis;
};

struct UnPackOption : public BaseOption {
    int number;
    int axis;
};

struct CastOption : public BaseOption {
    DataType in_data_type;
    DataType out_data_type;
};

struct GatherOption : public BaseOption {
    int axis;
    int batch_dims;
};

struct DepthToSpaceOption : public BaseOption {
    int block_size;
};

struct SpaceToDepthOption : public BaseOption {
    int block_size;
};

// For split and splitV
struct SplitOption : public BaseOption {
    int num_splits;
};

struct StridedSliceOption : public BaseOption {
    int  begin_mask;
    int  end_mask;
    int  ellipsis_mask;
    int  new_axis_mask;
    int  shrink_axis_mask;
    bool offset;
};

struct LeakyReLUOption : public BaseOption {
    float alpha;
};

struct GELUOption : public BaseOption {
    float approximate;
};

struct MirrorPadOption : public BaseOption {
    MirrorPadType mirror_pad_type;
};

struct OneHotOption : public BaseOption {
    int axis;
};

struct ShapeOption : public BaseOption {
    DataType out_type;
};

struct ArgMaxOption : public BaseOption {
    DataType out_type;
};

struct ArgMinOption : public BaseOption {
    DataType out_type;
};

struct BatchMatmulOption : public BaseOption {
    bool adj_x;
    bool adj_y;
    bool asym_quantize_inputs;
};

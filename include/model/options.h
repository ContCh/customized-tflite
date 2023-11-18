#pragma once

#include <vector>

#include "model/types.h"

struct BaseOption {};

/// Define necessary param type in option
enum Padding { SAME = 0, VALID = 1 };

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

struct ReshapeOption : public BaseOption {
    std::vector<int32_t> new_shape;
};

struct SoftmaxOption : public BaseOption {
    float beta;
};

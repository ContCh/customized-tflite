#ifndef CUSTOM_TFLITE_MODEL_OPTIONS_H
#define CUSTOM_TFLITE_MODEL_OPTIONS_H

#include <vector>

#include "model/types.h"

struct BaseOption {};

struct Conv2DOption : public BaseOption {
    int     stride_w;
    int     stride_h;
    int     dilation_w;
    int     dilation_h;
    Padding pad_type;
};

struct Pool2DOption : public BaseOption  {
    int     stride_w;
    int     stride_h;
    int     filter_w;
    int     filter_h;
    Padding pad_type;
};

struct DepthwiseConv2DOption : public BaseOption {
    int     stride_w;
    int     stride_h;
    int     dilation_w;
    int     dilation_h;
    int     depth_multiplier;
    Padding pad_type;
};

struct ReshapeOption : public BaseOption {
    std::vector<int32_t> new_shape;
};

struct SoftmaxOption : public BaseOption {
    float beta;
};


#endif // CUSTOM_TFLITE_MODEL_OPTIONS_H

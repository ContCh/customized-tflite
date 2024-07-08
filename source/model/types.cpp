#include "model/types.h"

#define ENUM_PRINT(EnumT, Val) \
    case EnumT::Val:           \
        return #Val;

// TODO: breakdown process if default. Something strange happens.
#define ENUM_DEFAULT_PRINT(Default) \
    default:                        \
        return "NonType";

std::string ToStr(OperatorType op_type) {
    switch (op_type) {
        ENUM_PRINT(OperatorType, NONE);
        ENUM_PRINT(OperatorType, ADD);
        ENUM_PRINT(OperatorType, AVERAGE_POOL);
        ENUM_PRINT(OperatorType, BATCH_NORMALIZATION);
        ENUM_PRINT(OperatorType, BATCH_TO_SPACE_ND);
        ENUM_PRINT(OperatorType, CONV2D);
        ENUM_PRINT(OperatorType, CONV3D);
        ENUM_PRINT(OperatorType, CONCAT);
        ENUM_PRINT(OperatorType, COSINE);
        ENUM_PRINT(OperatorType, DEPTHWISE_CONV2D);
        ENUM_PRINT(OperatorType, DEPTH_TO_SPACE);
        ENUM_PRINT(OperatorType, DEQUANTIZE);
        ENUM_PRINT(OperatorType, DIV);
        ENUM_PRINT(OperatorType, EXP);
        ENUM_PRINT(OperatorType, EXPAND_DIMS);
        ENUM_PRINT(OperatorType, FULLY_CONNECTED);
        ENUM_PRINT(OperatorType, SPACE_TO_DEPTH);
        ENUM_PRINT(OperatorType, L2_NORMALIZATION);
        ENUM_PRINT(OperatorType, L2_POOL);
        ENUM_PRINT(OperatorType, UNIDIRECTIONAL_LSTM);
        ENUM_PRINT(OperatorType, LOCAL_RESPONSE_NORMALIZATION);
        ENUM_PRINT(OperatorType, LOG);
        ENUM_PRINT(OperatorType, LOG_SOFTMAX);
        ENUM_PRINT(OperatorType, LOGISTIC);
        ENUM_PRINT(OperatorType, MAX_POOL);
        ENUM_PRINT(OperatorType, MUL);
        ENUM_PRINT(OperatorType, ONE_HOT);
        ENUM_PRINT(OperatorType, ReLU);
        ENUM_PRINT(OperatorType, ReLU1);
        ENUM_PRINT(OperatorType, ReLU6);
        ENUM_PRINT(OperatorType, PReLU);
        ENUM_PRINT(OperatorType, HARDSWISH);
        ENUM_PRINT(OperatorType, SOFTMAX);
        ENUM_PRINT(OperatorType, SUB);
        ENUM_PRINT(OperatorType, TANH);
        ENUM_PRINT(OperatorType, TRANSPOSE_CONV2D);
        ENUM_PRINT(OperatorType, CAST);
        ENUM_PRINT(OperatorType, GATHER);
        ENUM_PRINT(OperatorType, GATHER_ND);
        ENUM_PRINT(OperatorType, RESIZE_BILINEAR);
        ENUM_PRINT(OperatorType, SIN);
        ENUM_PRINT(OperatorType, PACK);
        ENUM_PRINT(OperatorType, PAD);
        ENUM_PRINT(OperatorType, PADV2);
        ENUM_PRINT(OperatorType, STRIDED_SLICE);
        ENUM_PRINT(OperatorType, SLICE);
        ENUM_PRINT(OperatorType, SQUEEZE);
        ENUM_PRINT(OperatorType, MEAN);
        ENUM_PRINT(OperatorType, ARGMAX);
        ENUM_PRINT(OperatorType, MAXIMUM);
        ENUM_PRINT(OperatorType, MINIMUM);
        ENUM_PRINT(OperatorType, BATCH_MATMUL);
        ENUM_PRINT(OperatorType, RESHAPE);
        ENUM_PRINT(OperatorType, RSQRT);
        ENUM_PRINT(OperatorType, SPACE_TO_BATCH_ND);
        ENUM_PRINT(OperatorType, SPLIT);
        ENUM_PRINT(OperatorType, SPLITV);
        ENUM_PRINT(OperatorType, SQRT);
        ENUM_PRINT(OperatorType, SQUARE);
        ENUM_PRINT(OperatorType, SQUARED_DIFFERENCE);
        ENUM_PRINT(OperatorType, SUM);
        ENUM_PRINT(OperatorType, TILE);
        ENUM_PRINT(OperatorType, TRANSPOSE);
        ENUM_PRINT(OperatorType, TOPK_V2);
        ENUM_PRINT(OperatorType, SELECT);
        ENUM_PRINT(OperatorType, SELECT_V2);
        ENUM_PRINT(OperatorType, EQUAL);
        ENUM_PRINT(OperatorType, NOT_EQUAL);
        ENUM_PRINT(OperatorType, POW);
        ENUM_PRINT(OperatorType, ARGMIN);
        ENUM_PRINT(OperatorType, UNPACK);
        ENUM_PRINT(OperatorType, RESIZE_NEAREST_NEIGHBOR);
        ENUM_PRINT(OperatorType, LEAKY_RELU);
        ENUM_PRINT(OperatorType, ABS);
        ENUM_PRINT(OperatorType, MIRROR_PAD);
        ENUM_PRINT(OperatorType, UNIQUE);
        ENUM_PRINT(OperatorType, UNIDIRECTIONAL_RNN);
        ENUM_PRINT(OperatorType, BIDIRECTIONAL_LSTM);
        ENUM_PRINT(OperatorType, REVERSEV2);
        ENUM_PRINT(OperatorType, BIDIRECTIONAL_RNN);
        ENUM_PRINT(OperatorType, ELU);
        ENUM_PRINT(OperatorType, REDUCE_MIN);
        ENUM_PRINT(OperatorType, REDUCE_MAX);
        ENUM_PRINT(OperatorType, REDUCE_PROD);
        ENUM_PRINT(OperatorType, REDUCE_ANY);
        ENUM_PRINT(OperatorType, REDUCE_ALL);
        ENUM_PRINT(OperatorType, WHERE);

        ENUM_DEFAULT_PRINT(NONE);
    }
}

std::string ToStr(DataType data_type) {
    switch (data_type) {
        ENUM_PRINT(DataType, UNDEFINED);
        ENUM_PRINT(DataType, INT4);
        ENUM_PRINT(DataType, UINT8);
        ENUM_PRINT(DataType, INT8);
        ENUM_PRINT(DataType, UINT16);
        ENUM_PRINT(DataType, INT16);
        ENUM_PRINT(DataType, UINT32);
        ENUM_PRINT(DataType, INT32);
        ENUM_PRINT(DataType, UINT64);
        ENUM_PRINT(DataType, INT64);
        ENUM_PRINT(DataType, FLOAT16);
        ENUM_PRINT(DataType, FLOAT32);
        ENUM_PRINT(DataType, FLOAT64);
        ENUM_PRINT(DataType, STRING);
        ENUM_PRINT(DataType, BOOL);

        ENUM_DEFAULT_PRINT(NONE);
    }
}

#undef ENUM_PRINT

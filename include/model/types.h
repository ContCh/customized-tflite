#ifndef CUSTOM_TFLITE_TYPES_H
#define CUSTOM_TFLITE_TYPES_H

#include <iostream>
#include <vector>

typedef uint32_t NODEID_T;
typedef uint32_t BLOBID_T;
typedef uint32_t BUFID_T;
static const uint32_t INVALID_ID = UINT32_MAX;

enum class OperatorType {
    NONE,
    // Math
    ADD,
    AVERAGE_POOL,
    BATCH_NORMALIZATION,
    CONV2D,
    CONCAT,
    COSINE,
    DEPTHWISE_CONV2D,
    DEPTH_TO_SPACE,
    DEQUANTIZE,
    DIV,
    EXP,
    EXPAND_DIMS,
    FULLY_CONNECTED,
    SPACE_TO_DEPTH,
    L2_NORMALIZATION,
    L2_POOL,
    UNIDIRECTIONAL_LSTM,
    LOCAL_RESPONSE_NORMALIZATION,
    LOG,
    LOGISTIC,
    MAX_POOL,
    MUL,
    ONE_HOT,
    ReLU,
    ReLU1,
    ReLU6,
    PReLU,
    HARDSWISH,
    SOFTMAX,
    SUB,
    TANH,
    TRANSPOSE_CONV2D,
    CAST,
    GATHER,
    RESIZE_BILINEAR,
    SIN,
    PACK,
    PAD,
    PADV2,
    STRIDED_SLICE,
    SLICE,
    SQUEEZE,
    MEAN,
    ARGMAX,
    MAXIMUM,
    MINIMUM,
    MATMUL,
    NEG,
    RESHAPE,
    RSQRT,
    SPLIT,
    SPLITV,
    SQRT,
    SQUARE,
    SQUARED_DIFFERENCE,
    SUM,
    TILE,
    TRANSPOSE,
    SELECT,
    SELECTV2,
    EQUAL,
    NOT_EQUAL,
    POW,
    ARGMIN,
    UNPACK,
    RESIZE_NEAREST_NEIGHBOR,
    LEAKY_RELU,
    ABS,
    MIRROR_PAD,
    UNIQUE,
    UNIDIRECTIONAL_RNN,
    BIDIRECTIONAL_LSTM,
    REVERSEV2,
    BIDIRECTIONAL_RNN,
    ELU,
    WHERE,
    ALL_OP_TYPES
};

enum class DataType {
    UNDEFINED = 0,
    // INT
    INT4,
    UINT8,
    INT8,
    UINT16,
    INT16,
    UINT32,
    INT32,
    UINT64,
    INT64,
    // FLOAT
    FLOAT16,
    FLOAT32,
    FLOAT64,
    // OTHER
    STRING,
    BOOL,
};

template <DataType T> struct DataTypeImpl {};

template <> struct DataTypeImpl<DataType::UNDEFINED> { typedef void Type; };
template <> struct DataTypeImpl<DataType::UINT8> { typedef uint8_t Type; };
template <> struct DataTypeImpl<DataType::INT8> { typedef int8_t Type; };
template <> struct DataTypeImpl<DataType::UINT16> { typedef uint16_t Type; };
template <> struct DataTypeImpl<DataType::INT16> { typedef int16_t Type; };
template <> struct DataTypeImpl<DataType::UINT32> { typedef uint32_t Type; };
template <> struct DataTypeImpl<DataType::INT32> { typedef int32_t Type; };
template <> struct DataTypeImpl<DataType::UINT64> { typedef uint64_t Type; };
template <> struct DataTypeImpl<DataType::INT64> { typedef int64_t Type; };
template <> struct DataTypeImpl<DataType::FLOAT16> { typedef uint16_t Type; };
template <> struct DataTypeImpl<DataType::FLOAT32> { typedef float Type; };
template <> struct DataTypeImpl<DataType::FLOAT64> { typedef double Type; };
template <> struct DataTypeImpl<DataType::BOOL> { typedef bool Type; };


#define ENUM_TYPE_PRINT_DECLARE(EnumT) \
std::ostream& operator<<(std::ostream&, EnumT)

ENUM_TYPE_PRINT_DECLARE(OperatorType);
ENUM_TYPE_PRINT_DECLARE(DataType);

#undef ENUM_TYPE_PRINT_DECLARE


#endif // CUSTOM_TFLITE_TYPES_H

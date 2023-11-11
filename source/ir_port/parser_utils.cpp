#include "ir_port/parser_utils.h"

#include <fcntl.h>
#include <map>

#include "common/logging.h"
#include <string>

namespace utils {
std::string GetContents(const std::string& model_path) {
    constexpr int STR_INIT_CAPABILITY = 2048;
    constexpr int READ_BUFFER_SIZE    = (1 << 16);
    std::string   file_contents;
    file_contents.reserve(STR_INIT_CAPABILITY);

    int fd = open(model_path.c_str(), O_RDONLY);
    REPORT_ERROR_IF(fd == -1, "Invalid model path, cannot access file: ", model_path);

    char read_buffer[READ_BUFFER_SIZE];
    while (true) {
        auto size = read(fd, read_buffer, READ_BUFFER_SIZE);
        if (size == -1) {
            close(fd);
            report_error("Error happen in reading file: ", model_path);
        }
        if (size == 0) {
            close(fd);
            break;
        }
        file_contents.append(read_buffer, size);
    }
    close(fd);
    return file_contents;
}

DataType GetMappedDataTypeOf(tflite::TensorType tensor_type) {
    switch (tensor_type) {
    case tflite::TensorType_INT8:
        return DataType::INT8;
    case tflite::TensorType_UINT8:
        return DataType::UINT8;
    case tflite::TensorType_INT16:
        return DataType::INT16;
    case tflite::TensorType_UINT16:
        return DataType::UINT16;
    case tflite::TensorType_INT32:
        return DataType::INT32;
    case tflite::TensorType_UINT32:
        return DataType::UINT32;
    case tflite::TensorType_FLOAT16:
        return DataType::FLOAT16;
    case tflite::TensorType_FLOAT32:
        return DataType::FLOAT32;
    case tflite::TensorType_FLOAT64:
        return DataType::FLOAT64;
    default:
        report_error("Unsupported tensor type.");
    }
    return DataType::UNDEFINED;
}

OperatorType GetMappedOpTypeOf(::tflite::BuiltinOperator op_code) {
    static std::map<tflite::BuiltinOperator, OperatorType> operator_type_map = {
        {tflite::BuiltinOperator_CONV_2D, OperatorType::CONV2D},
        {tflite::BuiltinOperator_DEPTHWISE_CONV_2D, OperatorType::DEPTHWISE_CONV2D},
        {tflite::BuiltinOperator_AVERAGE_POOL_2D, OperatorType::AVERAGE_POOL},
        {tflite::BuiltinOperator_MAX_POOL_2D, OperatorType::MAX_POOL},
        {tflite::BuiltinOperator_RESHAPE, OperatorType::RESHAPE},
        {tflite::BuiltinOperator_SOFTMAX, OperatorType::SOFTMAX}};
    return operator_type_map.at(op_code);
}

} // namespace utils

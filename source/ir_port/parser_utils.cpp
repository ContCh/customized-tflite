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


OperatorType GetMappedActTypeOf(::tflite::ActivationFunctionType act_type) {
    static std::map<tflite::ActivationFunctionType, OperatorType> act_type_map = {
        {tflite::ActivationFunctionType_NONE, OperatorType::NONE},
        {tflite::ActivationFunctionType_RELU, OperatorType::ReLU},
        {tflite::ActivationFunctionType_RELU6, OperatorType::ReLU6},
        {tflite::ActivationFunctionType_TANH, OperatorType::TANH}};
    return act_type_map.at(act_type);
}

Padding GetMappedPaddingOf(::tflite::Padding padding) {
    switch (padding) {
        case ::tflite::Padding_SAME:
            return Padding::SAME;
        case ::tflite::Padding_VALID:
            return Padding::VALID;
        default:
            return Padding::VALID;
    }
}

} // namespace utils

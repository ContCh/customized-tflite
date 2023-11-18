#pragma once

#include <string>
#include <vector>

#include "schema_generated.h"
#include "model/types.h"
#include "model/options.h"

namespace utils {
std::string GetContents(const std::string& model_path);

template <typename T>
std::vector<T> GetVecData(const ::flatbuffers::Vector<T>* flatbuffer_vec) {
    std::vector<T> data(flatbuffer_vec->size());
    for (size_t idx = 0; idx < data.size(); idx++) {
        data[idx] = flatbuffer_vec->Get(idx);
    }
    return data;
}

DataType GetMappedDataTypeOf(::tflite::TensorType tensor_type);

Padding GetMappedPaddingOf(::tflite::Padding padding);

OperatorType GetMappedActTypeOf(::tflite::ActivationFunctionType op_code);

} // namespace utils

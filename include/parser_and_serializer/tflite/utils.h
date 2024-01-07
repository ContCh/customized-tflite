#pragma once

#include <string>
#include <vector>

#include "model/options.h"
#include "model/types.h"
#include "schema_generated.h"

namespace utils {
std::string GetContents(const std::string& model_path);

template <typename T> std::vector<T> GetVecData(const ::flatbuffers::Vector<T>* flatbuffer_vec) {
    std::vector<T> data(flatbuffer_vec->size());
    for (size_t idx = 0; idx < data.size(); idx++) {
        data[idx] = flatbuffer_vec->Get(idx);
    }
    return data;
}

DataType           GetMappedDataTypeOf(::tflite::TensorType tensor_type);
tflite::TensorType GetMappedDataTypeOf(DataType data_type);

Padding           GetMappedPaddingOf(::tflite::Padding padding);
::tflite::Padding GetMappedPaddingOf(Padding padding);

OperatorType                     GetMappedActTypeOf(::tflite::ActivationFunctionType op_code);
::tflite::ActivationFunctionType GetMappedActTypeOf(OperatorType op_code);

}  // namespace utils

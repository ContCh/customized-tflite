#include "model/data_blob.h"

#include "common/logging.h"

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "Dim : [";
    for (auto dim : shape.GetDims()) {
        os << ' ' << dim;
    }
    os << " ]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const DataBlob& blob) {
    os << "DataBlob " << '(' << ToStr(blob.GetDataType()) << ") " << blob.GetShape();
    return os;
}

DataBlob::QuantParam& DataBlob::CreateQuantParam() {
    REPORT_ERROR_IF(quantization_params_ != nullptr, "Already create quant param!");
    quantization_params_ = std::make_unique<QuantParam>();
    return *quantization_params_;
}

#include "model/data_blob.h"

#include "common/logging.h"
#include "model/graph.h"

Operator* DataBlob::OperatorIterator::dereference() {
    auto* op = graph_.GetOperator(*base());
    REPORT_ERROR_IF(op == nullptr, "Fail to find operator in graph.");
    return op;
}

void DataBlob::SetProducer(const Operator* op) { producer_ = op->GetID(); }

Operator* DataBlob::GetProducer() const { return graph_.GetOperator(producer_); }

void DataBlob::AddConsumer(const Operator* op) { consumers_.push_back(op->GetID()); }

Range<DataBlob::OperatorIterator> DataBlob::GetConsumers() const {
    return Range<DataBlob::OperatorIterator> {
        {graph_, consumers_.begin()},
        {graph_, consumers_.end()  }
    };
}

DataBlob::QuantParam& DataBlob::CreateQuantParam() {
    REPORT_ERROR_IF(quantization_params_ != nullptr, "Already create quant param!");
    quantization_params_ = std::make_unique<QuantParam>();
    return *quantization_params_;
}

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

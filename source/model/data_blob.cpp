#include "model/data_blob.h"

std::ostream &operator<<(std::ostream &os, const Shape &shape) {
    os << "Dim : [";
    for (auto dim : shape.getDims()) {
        os << ' ' << dim;
    }
    os << " ]";
    return os;
}

std::ostream &operator<<(std::ostream &os, const DataBlob &blob) {
    os << "DataBlob " << '(' << blob.getDataType() << ") " << blob.getShape();
    return os;
}

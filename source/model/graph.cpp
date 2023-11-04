#include "model/graph.h"

Operator* Graph::GetOperator(NODEID_T node_id) {
    if (operators_.find(node_id) == operators_.end()) {
        return nullptr;
    }
    return operators_.at(node_id);
}

DataBlob* Graph::GetDataBlob(BLOBID_T blob_id) {
    if (data_blobs_.find(blob_id) == data_blobs_.end()) {
        return nullptr;
    }
    return data_blobs_.at(blob_id);
}

std::vector<uint8_t>* Graph::GetBuffer(BLOBID_T blob_id) {
    if (buffers_.find(blob_id) == buffers_.end()) {
        return nullptr;
    }
    return buffers_.at(blob_id);
}

Operator* Graph::AddOperator(OperatorType op_type) {
    auto* new_operator                 = new Operator(op_type);
    data_blobs_[new_operator->GetID()] = std::unique_ptr<Operator>(new_operator);
    return new_operator;
}

DataBlob* Graph::AddDataBlob(std::string name) {
    auto* new_data_blob                 = new DataBlob(name);
    data_blobs_[new_data_blob->GetID()] = std::unique_ptr<DataBlob>(new_data_blob);
    return new_data_blob;
}

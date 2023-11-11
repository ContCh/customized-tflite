#include "model/graph.h"


std::vector<Operator*> Graph::GetOperators() const {
    std::vector<Operator*> ops_list;
    std::transform(operators_.begin(), operators_.end(), std::back_inserter(ops_list),
                   [](const std::pair<const NODEID_T, std::unique_ptr<Operator>>& id_to_op) {
                       return id_to_op.second.get();
                   });
    return ops_list;
}

Operator* Graph::GetOperator(NODEID_T node_id) const {
    if (operators_.find(node_id) == operators_.end()) {
        return nullptr;
    }
    return operators_.at(node_id).get();
}

DataBlob* Graph::GetDataBlob(BLOBID_T blob_id) const {
    if (data_blobs_.find(blob_id) == data_blobs_.end()) {
        return nullptr;
    }
    return data_blobs_.at(blob_id).get();
}

const std::vector<uint8_t>* Graph::GetBuffer(BLOBID_T blob_id) const {
    if (buffers_.find(blob_id) == buffers_.end()) {
        return nullptr;
    }
    return &(buffers_.at(blob_id));
}

std::vector<uint8_t>* Graph::GetBuffer(BLOBID_T blob_id) {
    if (buffers_.find(blob_id) == buffers_.end()) {
        return nullptr;
    }
    return &(buffers_.at(blob_id));
}

Operator* Graph::AddOperator(OperatorType op_type) {
    auto* new_operator                = new Operator(op_type);
    operators_[new_operator->GetID()] = std::unique_ptr<Operator>(new_operator);
    return new_operator;
}

DataBlob* Graph::AddDataBlob(std::string name) {
    auto* new_data_blob                 = new DataBlob(name);
    data_blobs_[new_data_blob->GetID()] = std::unique_ptr<DataBlob>(new_data_blob);
    return new_data_blob;
}

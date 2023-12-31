#include "model/graph.h"

#include "common/stl_wrapper.h"

std::vector<Operator*> Graph::GetOperators() const {
    std::vector<Operator*> ops_list;
    common::transform(
        operators_, std::back_inserter(ops_list),
        [](const std::pair<const NODEID_T, std::unique_ptr<Operator>>& id_to_op) { return id_to_op.second.get(); });
    return ops_list;
}

std::vector<DataBlob*> Graph::GetDataBlobs() const {
    std::vector<DataBlob*> data_blobs_list;
    common::transform(
        data_blobs_, std::back_inserter(data_blobs_list),
        [](const std::pair<const BLOBID_T, std::unique_ptr<DataBlob>>& id_to_blob) { return id_to_blob.second.get(); });
    return data_blobs_list;
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

void Graph::EraseOperator(std::function<bool(const Operator*)> erase_cond) {
    for (auto op_it = operators_.begin(); op_it != operators_.end();) {
        if (erase_cond(op_it->second.get())) {
          op_it = operators_.erase(op_it);
            continue;
        }
        op_it++;
    }
}

void Graph::EraseBlob(std::function<bool(const DataBlob*)> erase_cond) {
    for (auto blob_it = data_blobs_.begin(); blob_it != data_blobs_.end();) {
        if (erase_cond(blob_it->second.get())) {
            blob_it = data_blobs_.erase(blob_it);
            continue;
        }
        blob_it++;
    }
}

#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common/logging.h"
#include "model/data_blob.h"
#include "model/operator.h"

class Graph {
 public:
    using BufferMap   = std::unordered_map<BLOBID_T, std::vector<uint8_t>>;
    using DataBlobMap = std::unordered_map<BLOBID_T, std::unique_ptr<DataBlob>>;
    using OperatorMap = std::unordered_map<NODEID_T, std::unique_ptr<Operator>>;

 public:
    std::vector<Operator*> GetOperators() const;
    std::vector<DataBlob*> GetDataBlobs() const;

    Operator*                   GetOperator(NODEID_T node_id) const;
    DataBlob*                   GetDataBlob(BLOBID_T blob_id) const;
    const std::vector<uint8_t>* GetBuffer(BLOBID_T blob_id) const;
    std::vector<uint8_t>*       GetBuffer(BLOBID_T blob_id);

    Operator* AddOperator(OperatorType op_type);
    DataBlob* AddDataBlob(std::string name);

    template <typename T>
    void SetBuffer(BLOBID_T blob_id, const std::vector<T>& buffer) {
        size_t bytes      = sizeof(T) * buffer.size();
        buffers_[blob_id] = std::vector<uint8_t>(bytes);
        memcpy(buffers_.at(blob_id).data(), buffer.data(), bytes);
    }

    void EraseOperator(Operator* op) { operators_.erase(op->GetID()); }

    const std::vector<BLOBID_T>& GetGraphInputs() const { return graph_inputs_; }
    const std::vector<BLOBID_T>& GetGraphOutputs() const { return graph_outputs_; }

    void SetGraphInputs(const std::vector<BLOBID_T>& inputs) { graph_inputs_ = inputs; }
    void SetGraphOutputs(const std::vector<BLOBID_T>& outputs) { graph_outputs_ = outputs; }

 private:
    DataBlobMap data_blobs_;
    OperatorMap operators_;
    BufferMap   buffers_;

    std::vector<BLOBID_T> graph_inputs_;
    std::vector<BLOBID_T> graph_outputs_;
};

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
        memcpy((void*)buffer.data(), (void*)buffers_.at(blob_id).data(), bytes);
    }

    void EraseOperator(Operator* op) { operators_.erase(op->GetID()); }

 private:
    DataBlobMap data_blobs_;
    OperatorMap operators_;
    BufferMap   buffers_;
};

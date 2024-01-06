#ifndef CUSTOM_TFLITE_OPERATOR_H
#define CUSTOM_TFLITE_OPERATOR_H

#include <memory>
#include <vector>

#include "common/id_generator.h"
#include "common/iterator_adaptor.h"
#include "common/logging.h"
#include "model/options.h"
#include "model/types.h"

class Graph;
class DataBlob;

class Operator {
 public:
    class DataBlobIterator
        : public IteratorAdaptor<DataBlobIterator, std::vector<BLOBID_T>::const_iterator, DataBlob*> {
     public:
        DataBlobIterator(const Graph& graph, const std::vector<BLOBID_T>::const_iterator& iter)
            : IteratorAdaptor(iter), graph_(graph) {}

        DataBlob* dereference();

     private:
        const Graph& graph_;
    };

 public:
    Operator(OperatorType op_type, const Graph& graph) : graph_(graph), operator_type_(op_type) {
        node_index_ = GenUniqueID();
    }
    ~Operator() {}

    NODEID_T     GetID() const { return node_index_; }
    OperatorType GetOpType() const { return operator_type_; }

    void AddInputBlob(const DataBlob* blob);
    void AddOutputBlob(const DataBlob* blob);

    Range<DataBlobIterator> GetInputBlobs() const;
    Range<DataBlobIterator> GetOutputBlobs() const;

    template <typename T> T* GetOption() {
        if (option_ == nullptr) {
            option_ = std::make_unique<T>();
        }
        return static_cast<T*>(option_.get());
    }

    template <typename T> const T* GetOption() const {
        REPORT_ERROR_IF(option_ == nullptr, "No available option.");
        return static_cast<const T*>(option_.get());
    }

 private:
    const Graph& graph_;
    NODEID_T     node_index_;
    OperatorType operator_type_ = OperatorType::NONE;

    std::vector<BLOBID_T>       inputs_;
    std::vector<BLOBID_T>       outputs_;
    std::unique_ptr<BaseOption> option_;
};

#endif  // CUSTOM_TFLITE_OPERATOR_H

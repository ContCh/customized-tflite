#ifndef CUSTOM_TFLITE_OPERATOR_H
#define CUSTOM_TFLITE_OPERATOR_H

#include <memory>
#include <vector>

#include "common/id_generator.h"
#include "model/options.h"
#include "model/types.h"

class Operator {
 public:
    Operator(OperatorType op_type) : operator_type_(op_type) { node_index_ = GenUniqueID(); }
    ~Operator() {}

    NODEID_T GetID() const { return node_index_; }

    void AddInput(BLOBID_T blob_id) { inputs_.push_back(blob_id); }
    void AddOutput(BLOBID_T blob_id) { outputs_.push_back(blob_id); }

    const std::vector<BLOBID_T>& GetInputs() const { return inputs_; }
    const std::vector<BLOBID_T>& GetOutputs() const { return outputs_; }

    OperatorType GetOpType() const { return operator_type_; }

    template <typename T>
    T* GetOption() {
        if (option_ == nullptr) {
            option_ = std::make_unique<T>();
        }
        return static_cast<T*>(option_.get());
    }

    template <typename T>
    const T* GetOption() const {
        REPORT_ERROR_IF(option_ == nullptr, "No available option.");
        return static_cast<const T*>(option_.get());
    }

 private:
    NODEID_T     node_index_;
    OperatorType operator_type_ = OperatorType::NONE;

    std::vector<BLOBID_T>       inputs_;
    std::vector<BLOBID_T>       outputs_;
    std::unique_ptr<BaseOption> option_;
};

#endif  // CUSTOM_TFLITE_OPERATOR_H

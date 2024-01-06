#pragma once

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "common/id_generator.h"
#include "common/iterator_adaptor.h"
#include "model/types.h"

class Shape {
 public:
    Shape() {}
    Shape(const std::vector<int>& dim_list) : dims_(dim_list) {}

    void SetDims(const std::vector<int>& dim_list) { dims_ = dim_list; }

    const std::vector<int>& GetDims() const { return dims_; }
    std::vector<int>&       MutableDims() { return dims_; }

    int GetDim(size_t i) const {
        // TODO: add check here avoid overflow
        return dims_[i];
    }

    bool operator==(const Shape& comp) const { return GetDims() == comp.GetDims(); }

    bool operator!=(const Shape& comp) const { return !((*this) == comp); }

    friend std::ostream& operator<<(std::ostream& os, const Shape& shape);

 private:
    // Should keep -1 or parse it to real value ? To keep -1, use int type.
    std::vector<int> dims_;
};

class Graph;
class Operator;

class DataBlob {
 public:
    struct QuantParam {
        float                max;
        float                min;
        std::vector<float>   scales;
        std::vector<int64_t> zero_points;
    };

    class OperatorIterator
        : public IteratorAdaptor<OperatorIterator, std::vector<BLOBID_T>::const_iterator, Operator*> {
     public:
        OperatorIterator(const Graph& graph, const std::vector<BLOBID_T>::const_iterator& iter)
            : IteratorAdaptor(iter), graph_(graph) {}

        Operator* dereference();

     private:
        const Graph& graph_;
    };

 public:
    DataBlob(std::string name, const Graph& graph) : graph_(graph), name_(name) { blob_index_ = GenUniqueID(); }

    BLOBID_T           GetID() const { return blob_index_; }
    const std::string& GetName() const { return name_; }

    void         SetShape(const Shape& shape) { blob_shape_ = shape; }
    Shape&       GetShape() { return blob_shape_; }
    const Shape& GetShape() const { return blob_shape_; }

    void     SetDataType(DataType data_type) { data_type_ = data_type; }
    DataType GetDataType() const { return data_type_; }

    void                    SetProducer(const Operator* op);
    Operator*               GetProducer() const;
    void                    AddConsumer(const Operator* op);
    Range<OperatorIterator> GetConsumers() const;

    QuantParam&       GetQuantParam() { return *quantization_params_; }
    const QuantParam& GetQuantParam() const { return *quantization_params_; }
    bool              HasQuantParam() const { return quantization_params_ != nullptr; }
    QuantParam&       CreateQuantParam();

    friend std::ostream& operator<<(std::ostream& os, const DataBlob& blob);

 private:
    const Graph&          graph_;
    BLOBID_T              blob_index_ = INVALID_ID;
    DataType              data_type_  = DataType::UNDEFINED;
    NODEID_T              producer_   = INVALID_ID;
    std::vector<NODEID_T> consumers_;

    Shape                       blob_shape_;
    std::unique_ptr<QuantParam> quantization_params_;
    std::string                 name_;
};

#include "model/operator.h"

#include "model/graph.h"

DataBlob* Operator::DataBlobIterator::dereference() {
    auto* data_blob = graph_.GetDataBlob(*base());
    REPORT_ERROR_IF(data_blob == nullptr, "Fail to find blob in graph.");
    return data_blob;
}

void Operator::AddInputBlob(const DataBlob* blob) { inputs_.push_back(blob->GetID()); }
void Operator::AddOutputBlob(const DataBlob* blob) { outputs_.push_back(blob->GetID()); }

Range<Operator::DataBlobIterator> Operator::GetInputBlobs() const {
    return Range<Operator::DataBlobIterator> {
        {graph_, inputs_.begin()},
        {graph_, inputs_.end()  }
    };
}

Range<Operator::DataBlobIterator> Operator::GetOutputBlobs() const {
    return Range<Operator::DataBlobIterator> {
        {graph_, outputs_.begin()},
        {graph_, outputs_.end()  }
    };
}

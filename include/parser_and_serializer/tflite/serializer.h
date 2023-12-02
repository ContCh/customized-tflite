#pragma once

#include "model/model.h"
#include "model/types.h"

#include "parser_and_serializer/tflite/op_resolver.h"

#include "flatbuffers/flatbuffers.h"
#include "schema_generated.h"

using flatbuffers::Offset;
using flatbuffers::Vector;

class TfLiteSerializer {
 public:
    void ExportToTfLite(const Model& model, std::string output_path);

 private:
    Offset<Vector<Offset<tflite::SubGraph>>>
    ExportSubGraphs(const Model& model, flatbuffers::FlatBufferBuilder* builder);

    Offset<Vector<Offset<tflite::Tensor>>> ExportTensors(const Graph&                    subgraph,
                                                         flatbuffers::FlatBufferBuilder* builder);

    Offset<Vector<Offset<tflite::Operator>>>
    ExportOperators(const Graph& subgraph, flatbuffers::FlatBufferBuilder* builder);

    Offset<Vector<Offset<tflite::OperatorCode>>>
    ExportOpCodes(const Model& model, flatbuffers::FlatBufferBuilder* builder);

    std::vector<Offset<tflite::Buffer>> ExportBuffers(const Model&                    model,
                                                      flatbuffers::FlatBufferBuilder* builder);

    OperatorResolver             op_resolver_;
    std::vector<OperatorType>    op_type_table_;
    std::map<BLOBID_T, uint32_t> data_blob_index_map_;
    std::map<BLOBID_T, uint32_t> buffer_index_map_;
};

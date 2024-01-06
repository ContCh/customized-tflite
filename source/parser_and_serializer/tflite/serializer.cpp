#include "parser_and_serializer/tflite/serializer.h"

#include <fstream>
#include <set>

#include "common/stl_wrapper.h"
#include "parser_and_serializer/tflite/utils.h"

#define TFLITE_SCHEMA_VERSION 3

void TfLiteSerializer::ExportToTfLite(const Model& model, std::string output_path) {
    LOG(INFO) << "TfLiteSerializer::ExportToTfLite Start.";
    flatbuffers::FlatBufferBuilder builder(10240);

    // Export op codes
    auto op_codes = ExportOpCodes(model, &builder);
    // Export buffers
    auto buffers = ExportBuffers(model, &builder);
    // Export SubGraphs
    auto subgraphs = ExportSubGraphs(model, &builder);
    // Export description
    auto description = builder.CreateString("custom_tflite repo export");
    // Export meta data
    Offset<tflite::Metadata> metadata =
        tflite::CreateMetadata(builder, builder.CreateString("metadata"), buffers.size());
    std::vector<Offset<tflite::Metadata>> metadatas = {metadata};

    auto tflite_model = CreateModel(builder, TFLITE_SCHEMA_VERSION, op_codes, subgraphs, description,
                                    builder.CreateVector(buffers), 0, builder.CreateVector(metadatas));
    ::tflite::FinishModelBuffer(builder, tflite_model);
    // Export to file
    const uint8_t* buffer = builder.GetBufferPointer();
    auto           size   = builder.GetSize();
    std::string    file_contents(reinterpret_cast<const char*>(buffer), size);
    std::ofstream  output_file(output_path, std::ios_base::out | std::ios_base::binary);
    if (!output_file.is_open()) {
        output_file.close();
        report_error("Cannot access or open file : ", output_path);
    }
    output_file << file_contents;
    output_file.close();
    LOG(INFO) << "TfLiteSerializer::ExportToTfLite End.";
}

Offset<Vector<Offset<tflite::SubGraph>>> TfLiteSerializer::ExportSubGraphs(const Model&                    model,
                                                                           flatbuffers::FlatBufferBuilder* builder) {
    const auto& main_graph = model.GetMainGraph();

    auto tensors   = ExportTensors(main_graph, builder);
    auto operators = ExportOperators(main_graph, builder);

    std::vector<int32_t> graph_inputs;
    for (auto input : main_graph.GetGraphInputs()) {
        graph_inputs.push_back(data_blob_index_map_.at(input));
    }
    std::vector<int32_t> graph_outputs;
    for (auto output : main_graph.GetGraphOutputs()) {
        graph_outputs.push_back(data_blob_index_map_.at(output));
    }

    auto subgraph = tflite::CreateSubGraph(*builder, tensors, builder->CreateVector(graph_inputs),
                                           builder->CreateVector(graph_outputs), operators, 0);
    std::vector<Offset<tflite::SubGraph>> subgraphs = {subgraph};
    return builder->CreateVector(subgraphs);
}

Offset<Vector<Offset<tflite::OperatorCode>>> TfLiteSerializer::ExportOpCodes(const Model&                    model,
                                                                             flatbuffers::FlatBufferBuilder* builder) {
    std::set<OperatorType> op_type_set;
    const auto&            main_graph = model.GetMainGraph();
    for (const auto* op : main_graph.GetOperators()) {
        op_type_set.insert(op->GetOpType());
    }
    common::copy(op_type_set, std::back_inserter(op_type_table_));

    std::vector<Offset<tflite::OperatorCode>> op_codes_vec;
    common::transform(op_type_table_, std::back_inserter(op_codes_vec), [&](OperatorType op_type) {
        auto   builtin_op_code = op_resolver_.GetMappedOpTypeOf(op_type);
        int8_t deprecated_builtin_code =
            std::min(static_cast<int32_t>(builtin_op_code),
                     static_cast<int32_t>(tflite::BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES));
        Offset<flatbuffers::String> custom_code = 0;
        const int32_t               version     = 2;
        return tflite::CreateOperatorCode(*builder, deprecated_builtin_code, custom_code, version, builtin_op_code);
    });
    return builder->CreateVector(op_codes_vec);
}

std::vector<Offset<tflite::Buffer>> TfLiteSerializer::ExportBuffers(const Model&                    model,
                                                                    flatbuffers::FlatBufferBuilder* builder) {
    std::vector<Offset<tflite::Buffer>> buffers;
    // Insert an empty buffer to the beginning of the list.
    buffers.push_back(tflite::CreateBuffer(*builder, 0));

    const auto& main_graph = model.GetMainGraph();
    auto        data_blobs = main_graph.GetDataBlobs();
    buffers.reserve(buffers.size() + data_blobs.size());
    for (const auto* blob : data_blobs) {
        const auto* buffer_ptr = main_graph.GetBuffer(blob->GetID());
        if (buffer_ptr == nullptr) {
            buffers.push_back(tflite::CreateBuffer(*builder, 0));
        } else {
            buffers.push_back(tflite::CreateBuffer(*builder, builder->CreateVector(*buffer_ptr)));
        }
        buffer_index_map_[blob->GetID()] = buffers.size() - 1;
    }
    return buffers;
}

Offset<Vector<Offset<tflite::Tensor>>> TfLiteSerializer::ExportTensors(const Graph&                    subgraph,
                                                                       flatbuffers::FlatBufferBuilder* builder) {
    data_blob_index_map_.clear();
    std::vector<DataBlob*> data_blobs(subgraph.GetDataBlobs().size());
    common::transform(subgraph.GetDataBlobs(), data_blobs.begin(), [](DataBlob* blob) { return blob; });
    std::sort(data_blobs.begin(), data_blobs.end(),
              [](const DataBlob* blob1, const DataBlob* blob2) { return blob1->GetName() < blob2->GetName(); });

    for (uint32_t index = 0; index < data_blobs.size(); index++) {
        data_blob_index_map_[data_blobs.at(index)->GetID()] = index;
    }

    std::vector<Offset<tflite::Tensor>> tensors;
    tensors.reserve(data_blobs.size());
    for (const auto* data_blob : data_blobs) {
        Offset<tflite::QuantizationParameters> quant_param = tflite::CreateQuantizationParameters(*builder);
        if (data_blob->HasQuantParam()) {
            auto                    quantization = data_blob->GetQuantParam();
            Offset<Vector<float>>   min;
            Offset<Vector<float>>   max;
            Offset<Vector<float>>   scale;
            Offset<Vector<int64_t>> zero_point;
            min         = builder->CreateVector(std::vector<float> {quantization.min});
            max         = builder->CreateVector(std::vector<float> {quantization.max});
            scale       = builder->CreateVector(quantization.scales);
            zero_point  = builder->CreateVector(quantization.zero_points);
            quant_param = tflite::CreateQuantizationParameters(*builder, min, max, scale, zero_point);
        }

        auto tensor_type = utils::GetMappedDataTypeOf(data_blob->GetDataType());
        auto shape       = data_blob->GetShape().GetDims();
        auto tensor      = tflite::CreateTensor(*builder, builder->CreateVector(shape), tensor_type,
                                                buffer_index_map_.at(data_blob->GetID()),
                                                builder->CreateString(data_blob->GetName()), quant_param, false);
        tensors.push_back(tensor);
    }

    return builder->CreateVector(tensors);
}

Offset<Vector<Offset<tflite::Operator>>> TfLiteSerializer::ExportOperators(const Graph&                    subgraph,
                                                                           flatbuffers::FlatBufferBuilder* builder) {
    auto ops = subgraph.GetOperators();

    std::vector<Offset<tflite::Operator>> tflite_ops;
    tflite_ops.reserve(ops.size());
    for (auto* op : ops) {
        uint32_t op_index = common::get_first_index(op_type_table_, op->GetOpType());
        REPORT_ERROR_IF(op_index >= op_type_table_.size(), "Op type is not registered when export");

        std::vector<int32_t> inputs;
        common::transform(op->GetInputBlobs(), std::back_inserter(inputs),
                          [&](const DataBlob* blob) { return data_blob_index_map_.at(blob->GetID()); });
        std::vector<int32_t> outputs;
        common::transform(op->GetOutputBlobs(), std::back_inserter(outputs),
                          [&](const DataBlob* blob) { return data_blob_index_map_.at(blob->GetID()); });
        auto  option_type = op_resolver_.GetMappedOptionTypeOf(op->GetOpType());
        auto* op_resolver = op_resolver_.GetOptionResolver(op->GetOpType());
        auto  op_option   = op_resolver->SerializeOption(*op, builder);

        tflite_ops.push_back(tflite::CreateOperator(*builder, op_index, builder->CreateVector(inputs),
                                                    builder->CreateVector(outputs), option_type, op_option, 0,
                                                    ::tflite::CustomOptionsFormat_FLEXBUFFERS));
    }
    return builder->CreateVector(tflite_ops);
}

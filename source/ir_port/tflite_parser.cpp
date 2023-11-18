#include <map>
#include <string.h>

#include "common/stl_wrapper.h"
#include "ir_port/parser_utils.h"
#include "ir_port/tflite_parser.h"

std::unique_ptr<Model> TfLiteParser::ImportModel(const std::string& tflite_file_path) {
    auto                 input_file_contents = utils::GetContents(tflite_file_path);
    const tflite::Model* input_model         = tflite::GetModel(input_file_contents.data());

    // Full list of all known operators.
    std::unique_ptr<Model> model = std::make_unique<Model>();
    // load tensors
    LoadTensors(*input_model, model.get());
    // operator table
    LoadOperatorsTable(*input_model);
    // load operators
    LoadOperators(*input_model, model.get());

    return model;
}

void TfLiteParser::LoadTensors(const tflite::Model& input_model, Model* model) {
    auto tensors = (*input_model.subgraphs())[0]->tensors();
    if (!tensors) {
        return;
    }

    auto* buffers    = input_model.buffers();
    auto& main_graph = model->GetMainGraph();
    for (const auto* tensor : *tensors) {
        auto* data_blob = main_graph.AddDataBlob(tensor->name()->c_str());
        data_blob_table_.push_back(data_blob);
        data_blob->SetDataType(utils::GetMappedDataTypeOf(tensor->type()));
        // Get buffer which contains data
        int   buffer_index = tensor->buffer();
        auto* src_buffer   = buffers->Get(buffer_index)->data();
        if (src_buffer != nullptr) {
            std::vector<uint8_t> src_data(src_buffer->data(),
                                          src_buffer->data() + src_buffer->size());
            main_graph.SetBuffer(data_blob->GetID(), src_data);
        }
        // Get blob shape
        auto shape = tensor->shape();
        if (shape != nullptr) {
            auto dimensions = utils::GetVecData(tensor->shape());
            data_blob->SetShape(Shape(dimensions));
        } else {
            // Set shape=1 by default
            auto dummy_shape = Shape(std::vector<int>{1});
            data_blob->SetShape(dummy_shape);
        }
        // Get quant param
        auto quantization = tensor->quantization();
        if (quantization != nullptr) {
            auto& quantization_param = data_blob->CreateQuantParam();
            if (quantization->min() != nullptr && quantization->max() != nullptr) {
                quantization_param.min = quantization->min()->Get(0);
                quantization_param.max = quantization->max()->Get(0);
            }
            if (quantization->scale() != nullptr && quantization->zero_point() != nullptr) {
                std::vector<float>   q_scales;
                std::vector<int64_t> q_zero_points;
                quantization_param.scales      = utils::GetVecData(quantization->scale());
                quantization_param.zero_points = utils::GetVecData(quantization->zero_point());
            }
        }
    }
}

void TfLiteParser::LoadOperatorsTable(const tflite::Model& input_model) {
    auto opcodes = input_model.operator_codes();
    if (opcodes == nullptr) {
        return;
    }
    for (const auto* opcode : *opcodes) {
        auto builtin_code =
            std::max(opcode->builtin_code(),
                     static_cast<tflite::BuiltinOperator>(opcode->deprecated_builtin_code()));
        // TODO: Support to parse customized op
        REPORT_ERROR_IF(builtin_code == tflite::BuiltinOperator_CUSTOM,
                        "Customized op is not supported.");
        op_type_table_.push_back(op_resolver_.GetMappedOpTypeOf(builtin_code));
    }
}

void TfLiteParser::LoadOperators(const tflite::Model& input_model, Model* model) {
    auto tflite_ops = (*input_model.subgraphs())[0]->operators();
    if (tflite_ops == nullptr) {
        return;
    }

    auto& main_graph = model->GetMainGraph();
    for (const auto* tflite_op : *tflite_ops) {
        uint32_t index       = tflite_op->opcode_index();
        auto     op_type     = op_type_table_.at(index);
        auto*    new_op      = main_graph.AddOperator(op_type);
        auto*    op_resolver = op_resolver_.GetOptionResolver(op_type);
        op_resolver->ParseOption(tflite_op->builtin_options(), *new_op);

        // TODO: Parse option for each type of operator
        auto inputs = utils::GetVecData(tflite_op->inputs());
        common::for_each(inputs, [&](int32_t input_idx) {
            auto* data_blob = data_blob_table_.at(input_idx);
            new_op->AddInput(data_blob->GetID());
            data_blob->AddConsumerID(new_op->GetID());
        });
        auto outputs = utils::GetVecData(tflite_op->outputs());
        common::for_each(outputs, [&](int32_t output_idx) {
            auto* data_blob = data_blob_table_.at(output_idx);
            new_op->AddOutput(data_blob->GetID());
            REPORT_ERROR_IF(data_blob->GetProducerID() != INVALID_ID, data_blob->GetName(),
                            " is the output of more than 2 ops, which is abnormal situation.");
            data_blob->SetProducerID(new_op->GetID());
        });
    }
}

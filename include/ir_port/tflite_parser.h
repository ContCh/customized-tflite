#pragma once

#include "model/model.h"
#include "schema_generated.h"

class TfLiteParser {
 public:
    std::unique_ptr<Model> ImportModel(const std::string& tflite_file_path);

 private:
    void LoadOperators(const tflite::Model& input_model, Model* model);

    void LoadOperatorsTable(const tflite::Model& input_model);

    void LoadTensors(const tflite::Model& input_model, Model* model);

    std::vector<OperatorType> op_type_table_;
    std::vector<DataBlob*>    data_blob_table_;
};

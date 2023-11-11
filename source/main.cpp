#include <iostream>

#include "model/model.h"
#include "ir_port/tflite_parser.h"
#include "ir_port/graphviz_serializer.h"


int main() {
    LogSettings::GetInstance()->SetMinimumLogLevel(0);
    std::string path = "/home/chenchong/Workspace/tflite_models/mobilenet_v1_quant.tflite";
    std::string file_contents = "";
    TfLiteParser tflite_parser;
    auto unique_model = tflite_parser.ImportModel(path);
    std::cout << unique_model.get() << std::endl;

    GraphvizSerializer to_graphviz;
    to_graphviz.ExportToGraphviz(*unique_model.get(), "./graph_viz.txt");

    return 0;
}

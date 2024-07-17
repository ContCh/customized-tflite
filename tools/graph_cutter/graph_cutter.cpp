#include "common/command_line_parser.h"
#include "common/string_utils.h"
#include "graph_cutter/cutting_utils.h"
#include "model/model.h"
#include "parser_and_serializer/tflite/parser.h"
#include "parser_and_serializer/tflite/serializer.h"

struct CutterOptions {
    Option<std::string> input_tflite_file;
    Option<std::string> output_tflite_file;
    Option<std::string> input_tensors;
    Option<std::string> output_tensors;
};

int main(int argc, char** argv) {
    CutterOptions     cutter_options;
    std::vector<Flag> flags = {
        Flag("--input_tflite", "-i", cutter_options.input_tflite_file, REQUIRED::YES,
             "The path of input tflite model, the target file which is going to be cut."),
        Flag("--output_tflite", "-o", cutter_options.output_tflite_file, REQUIRED::YES,
             "The output path of tflite model which is processed. Specify the file path to save cut model."),
        Flag("--from", "-f", cutter_options.input_tensors, REQUIRED::YES,
             "The start tensors of cutting graph, as new inputs of processed model. If there are multiple tensors,"
             "use \',\' to seperate tensors name."),
        Flag("--to", "-t", cutter_options.output_tensors, REQUIRED::YES,
             "The end tensors of cutting graph, as new outputs of processed model. If there are multiple tensors,"
             "use \',\' to seperate tensors name."),
    };
    CommandLineParser::Parse(argc, argv, flags);

    auto model          = TfLiteParser().ImportModel(cutter_options.input_tflite_file.GetValue());
    auto input_tensors  = common::split(cutter_options.input_tensors.GetValue(), ',');
    auto output_tensors = common::split(cutter_options.output_tensors.GetValue(), ',');
    CuttingUtils::CutGraphImpl(*model.get(), input_tensors, output_tensors);
    TfLiteSerializer().ExportToTfLite(*model.get(), cutter_options.output_tflite_file.GetValue());

    return 0;
}

#include "parser_and_serializer/graphviz/serializer.h"

#include <fstream>

#include "model/types.h"

void GraphvizSerializer::ExportToGraphviz(const Model& model, std::string output_path) {
    LOG(INFO) << "GraphvizSerializer::ExportToGraphviz Start.";
    std::ofstream output_file(output_path.c_str(), std::ios::out);
    REPORT_ERROR_IF(!output_file.is_open(), "Failed to open file ", output_path);
    auto& main_graph = model.GetMainGraph();
    output_file << "digraph G {\n";
    output_file << "  node [shape=plaintext]\n";

    ExportOperators(main_graph, output_file);
    ExportBlobsAndEdges(main_graph, output_file);
    output_file << "}\n";
    output_file.close();
    LOG(INFO) << "GraphvizSerializer::ExportToGraphviz End.";
}

void GraphvizSerializer::ExportOperators(const Graph& subgraph, std::ostream& output_stream) {
    DLOG(INFO) << "GraphvizSerializer::ExportOperators Start.";
    for (const auto* op : subgraph.GetOperators()) {
        std::string op_type_str       = ToStr(op->GetOpType());
        std::string op_viz_name       = op_type_str + '_' + std::to_string(op->GetID());
        op_viz_name_map_[op->GetID()] = op_viz_name;
        output_stream << "  " << op_viz_name;
        output_stream << R"([label=<<TABLE BORDER="1" CELLBORDER="0" BGCOLOR="cyan4">)";
        output_stream << R"(<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="16">[ID:)" << op->GetID() << "] " << op_type_str
                      << R"(</FONT></TD></TR>)";
        // TODO: export option for each type of operator
        output_stream << R"(</TABLE>>])";
        output_stream << '\n';
    }
}

void GraphvizSerializer::ExportBlobsAndEdges(const Graph& subgraph, std::ostream& output_stream) {
    DLOG(INFO) << "GraphvizSerializer::ExportBlobsAndEdges Start.";
    for (const auto& [op_id, op_viz_name] : op_viz_name_map_) {
        const auto* op = subgraph.GetOperator(op_id);
        for (auto* data_blob : op->GetInputBlobs()) {
            if (subgraph.GetBuffer(data_blob->GetID()) != nullptr) {
                continue;
            }
            auto blob_viz_name = ExportBlob(data_blob, output_stream);
            output_stream << "  " << blob_viz_name << "->" << op_viz_name << '\n';
        }
        for (auto* data_blob : op->GetOutputBlobs()) {
            auto blob_viz_name = ExportBlob(data_blob, output_stream);
            output_stream << "  " << op_viz_name << "->" << blob_viz_name << '\n';
        }
    }
}

std::string GraphvizSerializer::ExportBlob(const DataBlob* data_blob, std::ostream& output_stream) {
    DLOG(INFO) << "GraphvizSerializer::ExportBlob Start.";
    std::string blob_viz_name = "BLOB_" + std::to_string(data_blob->GetID());
    output_stream << "  " << blob_viz_name;
    output_stream << R"([label=<<TABLE BORDER="1" CELLBORDER="0" BGCOLOR="tan1">)";
    output_stream << R"(<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="16">[BLOB_ID:)" << data_blob->GetID()
                  << R"(] </FONT></TD></TR>)";
    output_stream << R"(<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="14">DATA_TYPE: )" << ToStr(data_blob->GetDataType())
                  << R"(</FONT></TD></TR>)";
    output_stream << R"(<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="14">)" << data_blob->GetShape()
                  << R"(</FONT></TD></TR>)";
    if (data_blob->HasQuantParam()) {
        auto quant_param = data_blob->GetQuantParam();
        output_stream << R"(<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="14"> SCALE: )" << quant_param.scales[0]
                      << "  ZERO_POINT: " << quant_param.zero_points[0] << R"(</FONT></TD></TR>)";
    }
    output_stream << R"(</TABLE>>])";
    output_stream << '\n';
    return blob_viz_name;
}

#pragma once

#include "model/model.h"

#include <fstream>
#include <map>

/**
 * GraphvizSerializer collect model's information and visualize the information
 * It describes the model structure by graphviz format, and user can type the command
 * "dot -Tsvg `output_path` -O" to generate the visualized model structure.
 * About graphviz, please refer to https://www.graphviz.org/ for more details.
 * We apply html-like grammar to make visualize more vivid and abstracting. The html-like
 * feature of graphviz can be found at https://www.graphviz.org/doc/info/shapes.html#html
 */
class GraphvizSerializer {
 public:
    void ExportToGraphviz(const Model& model, std::string output_path);

 private:
    void        ExportOperators(const Graph& subgraph, std::ostream& output_stream);
    void        ExportBlobsAndEdges(const Graph& subgraph, std::ostream& output_stream);
    std::string ExportBlob(const DataBlob* data_blob, std::ostream& output_stream);

    std::map<NODEID_T, std::string> op_viz_name_map_;
};

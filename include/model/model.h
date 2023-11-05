#pragma once

#include "model/graph.h"

struct ModelFlags {};

class Model {
 public:
    Graph&       GetMainGraph() { return graph_; }
    const Graph& GetMainGraph() const { return graph_; }

    ModelFlags&       GetModelFlags() { return flags_; };
    const ModelFlags& GetModelFlags() const { return flags_; };

 private:
    ModelFlags flags_;
    Graph      graph_;
};

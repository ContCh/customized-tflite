#include <set>
#include <vector>

#include "model/model.h"

class CuttingUtils {
 public:
    static void CutGraphImpl(Model&                          model,
                             const std::vector<std::string>& input_tensors,
                             const std::vector<std::string>& output_tensors);

 private:
    static std::set<const Operator*> CollectOpsForward(const Graph& graph, const std::vector<BLOBID_T>&);

    static std::set<const Operator*> CollectOpsBackward(const Graph& graph, const std::vector<BLOBID_T>&);

    static void RemoveUnnecessaryOpsAndBlobs(Graph& graph, const std::set<const Operator*>& ops_to_keep);
};

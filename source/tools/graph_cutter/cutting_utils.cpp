#include "tools/graph_cutter/cutting_utils.h"

#include <queue>

#include "common/stl_wrapper.h"

void CuttingUtils::CutGraphImpl(Model&                          model,
                                const std::vector<std::string>& input_tensors,
                                const std::vector<std::string>& output_tensors) {
    auto& main_graph = model.GetMainGraph();

    std::vector<BLOBID_T> sub_inputs(input_tensors.size(), INVALID_ID);
    std::vector<BLOBID_T> sub_outputs(output_tensors.size(), INVALID_ID);
    for (const auto* blob : main_graph.GetDataBlobs()) {
        if (common::contains(input_tensors, blob->GetName())) {
            auto location        = common::get_first_index(input_tensors, blob->GetName());
            sub_inputs[location] = blob->GetID();
        }
        if (common::contains(output_tensors, blob->GetName())) {
            auto location         = common::get_first_index(output_tensors, blob->GetName());
            sub_outputs[location] = blob->GetID();
        }
    }
    if (common::contains(sub_inputs, INVALID_ID)) {
        auto input_tensor_name = input_tensors.at(common::get_first_index(sub_inputs, INVALID_ID));
        REPORT_ERROR_IF(std::count(input_tensors.begin(), input_tensors.end(), input_tensor_name) > 1, "`",
                        input_tensor_name, "` is duplicated. Please check arguments.");
        report_error("`", input_tensor_name, "` doesn't exist. Please check tensor's name.")
    }
    if (common::contains(sub_outputs, INVALID_ID)) {
        auto output_tensor_name = input_tensors.at(common::get_first_index(sub_outputs, INVALID_ID));
        REPORT_ERROR_IF(std::count(output_tensors.begin(), output_tensors.end(), output_tensor_name) > 1, "`",
                        output_tensor_name, "` is duplicated. Please check arguments.");
        report_error("`", output_tensor_name, "` doesn't exist. Please check tensor's name.")
    }
    auto forward_op_set  = CollectOpsForward(main_graph, sub_inputs);
    auto backward_op_set = CollectOpsBackward(main_graph, sub_outputs);
    std::set<const Operator*> ops_to_keep;
    for (const auto* op : forward_op_set) {
        if (common::contains(backward_op_set, op)) {
            ops_to_keep.insert(op);
        }
    }
    REPORT_ERROR_IF(ops_to_keep.empty(), "No operators or blobs will be kept, please check inputs and outputs.");
    RemoveUnnecessaryOpsAndBlobs(main_graph, ops_to_keep);
    main_graph.SetGraphInputs(sub_inputs);
    main_graph.SetGraphOutputs(sub_outputs);
}

std::set<const Operator*> CuttingUtils::CollectOpsForward(const Graph& graph, const std::vector<BLOBID_T>& blob_ids) {
    std::queue<const Operator*> ops_queue;
    std::set<const Operator*>   ops_traverse;
    for (auto blob_id : blob_ids) {
        auto* blob = graph.GetDataBlob(blob_id);
        for (auto op_id : blob->GetConsumerIDs()) {
            if (ops_traverse.count(graph.GetOperator(op_id))) {
                continue;
            }
            ops_traverse.insert(graph.GetOperator(op_id));
            ops_queue.push(graph.GetOperator(op_id));
        }
    }

    while (!ops_queue.empty()) {
        const auto* op = ops_queue.front();
        ops_queue.pop();
        for (auto blob_id : op->GetOutputs()) {
            auto* blob = graph.GetDataBlob(blob_id);
            for (auto op_id : blob->GetConsumerIDs()) {
                if (ops_traverse.count(graph.GetOperator(op_id))) {
                    continue;
                }
                ops_traverse.insert(graph.GetOperator(op_id));
                ops_queue.push(graph.GetOperator(op_id));
            }
        }
    }
    return ops_traverse;
}

std::set<const Operator*> CuttingUtils::CollectOpsBackward(const Graph& graph, const std::vector<BLOBID_T>& blob_ids) {
    std::queue<const Operator*> ops_queue;
    std::set<const Operator*>   ops_traverse;
    for (auto blob_id : blob_ids) {
        auto* blob  = graph.GetDataBlob(blob_id);
        auto  op_id = blob->GetProducerID();
        if (op_id == INVALID_ID) {
            continue;
        }
        ops_traverse.insert(graph.GetOperator(op_id));
        ops_queue.push(graph.GetOperator(op_id));
    }
    while (!ops_queue.empty()) {
        const auto* op = ops_queue.front();
        ops_queue.pop();
        for (auto blob_id : op->GetInputs()) {
            auto* blob  = graph.GetDataBlob(blob_id);
            auto  op_id = blob->GetProducerID();
            if (op_id == INVALID_ID) {
                continue;
            }
            ops_traverse.insert(graph.GetOperator(op_id));
            ops_queue.push(graph.GetOperator(op_id));
        }
    }
    return ops_traverse;
}

void CuttingUtils::RemoveUnnecessaryOpsAndBlobs(Graph& graph, const std::set<const Operator*>& ops_to_keep) {
    std::set<const DataBlob*> blobs_to_keep;
    for (const auto* op : ops_to_keep) {
        for (auto blob_id : op->GetInputs()) {
            blobs_to_keep.insert(graph.GetDataBlob(blob_id));
        }
        for (auto blob_id : op->GetOutputs()) {
            blobs_to_keep.insert(graph.GetDataBlob(blob_id));
        }
    }
    // Remove
    graph.EraseBlob([&](const DataBlob* blob) { return !common::contains(blobs_to_keep, blob); });
    graph.EraseOperator([&](const Operator* op) { return !common::contains(ops_to_keep, op); });
}

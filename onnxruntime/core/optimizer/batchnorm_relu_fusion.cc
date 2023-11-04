// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/batchnorm_relu_fusion.h"
#include "core/optimizer/initializer.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status BatchnormReluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  LOGS_DEFAULT(INFO) << "Called into fuser for BatchNorm + Relu";
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node = graph.GetNode(index);
    // check that node hasn't already been removed
    if (!node)
      continue;

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "BatchNormalization", {1, 14}) ||
        !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
        node->GetOutputEdgesCount() != 1) { 
      continue;
    }

    const auto& next_node = *(node->OutputNodesBegin());

    if (next_node.GetExecutionProviderType() != node->GetExecutionProviderType()) {
      // std::cout << "@2" <<std::endl;
      continue;
    }

    // if (!graph.GetNodeOutputsInGraphOutputs(*node).empty() || !graph.GetNodeOutputsInGraphOutputs(next_node).empty()) {
    if (!graph.GetNodeOutputsInGraphOutputs(*node).empty()) {
      // std::cout << "@3" <<std::endl;
      continue;
    }

    if (node->GetExecutionProviderType() == onnxruntime::kCpuExecutionProvider) {
      // Test if this is an activation that can be fused and also extract the
      // activation's parameters.
      // std::cout << "@5" <<std::endl;
      std::vector<float> activation_params;
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Relu", {6, 14})) {
        // std::cout << "@6" <<std::endl;
        continue;
      }
      // if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Relu", {6, 13}) &&
      //     !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Sigmoid", {6, 13}) &&
      //     !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Tanh", {6, 13})) {
      //   if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "LeakyRelu", {6})) {
      //     activation_params.push_back(graph_utils::GetNodeAttribute(next_node, "alpha")->f());
      //   } else if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Clip", {6, 11, 12, 13})) {
      //     float min, max;
      //     if (GetClipConstantMinMax(graph, next_node, min, max)) {
      //       activation_params.push_back(min);
      //       activation_params.push_back(max);
      //     } else {
      //       continue;
      //     }
      //   } else {
      //     continue;
      //   }
      // }

      Node& batchnorm_node = *node;
      Node& act_node = *graph.GetNode(next_node.Index());

      Node& fused_batchnorm = graph.AddNode(graph.GenerateNodeName("fused " + batchnorm_node.Name()), "FusedBatchNorm",
                                       "fused BatchNorm " + batchnorm_node.Name() + "with activation " + act_node.OpType(),
                                       batchnorm_node.MutableInputDefs(),
                                       {},
                                       &batchnorm_node.GetAttributes(),
                                       "");

      // Assign provider to this new node. Provider should be same as the provider for old node.
      fused_batchnorm.SetExecutionProviderType(batchnorm_node.GetExecutionProviderType());

      // Add attributes to specify the activation type and parameters.
      fused_batchnorm.AddAttribute("activation", next_node.OpType());
      if (activation_params.size() > 0) {
        fused_batchnorm.AddAttribute("activation_params", activation_params);
      }

      // move output definitions and edges from act_node to fused_batchnorm. delete batchnorm_node and act_node.
      graph_utils::FinalizeNodeFusion(graph, {batchnorm_node, act_node}, fused_batchnorm);

      modified = true;
    } else {
      // std::cout << "@4" <<std::endl;
      continue;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime

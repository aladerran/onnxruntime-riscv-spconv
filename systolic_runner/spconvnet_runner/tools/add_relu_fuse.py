import onnx
from onnx import helper

def fuse_systolic_add_relu(model):
    nodes_to_remove = []
    nodes_to_add = []

    for node in model.graph.node:
        if node.op_type == "SystolicAdd":
            for child_node in model.graph.node:
                if child_node.op_type == "Relu" and child_node.input[0] == node.output[0]:
                    fused_node = helper.make_node(
                        'SystolicAddRelu', 
                        inputs=node.input, 
                        outputs=child_node.output, 
                        name=node.name + '_SystolicAddRelu' 
                    )
                    nodes_to_add.append(fused_node)
                    nodes_to_remove.append(node)
                    nodes_to_remove.append(child_node)
                    break

    for node in nodes_to_remove:
        model.graph.node.remove(node)

    model.graph.node.extend(nodes_to_add)

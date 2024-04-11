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

def fuse_systolic_bn_relu(model):
    nodes_to_remove = []
    nodes_to_add = []

    for node in model.graph.node:
        if node.op_type == "SystolicBatchNorm":
            for child_node in model.graph.node:
                if child_node.op_type == "Relu" and child_node.input[0] == node.output[0]:
                    fused_node = helper.make_node(
                        'SystolicBatchNormRelu',
                        inputs=node.input,
                        outputs=child_node.output,
                        name=node.name + '_SystolicBatchNormRelu'
                    )
                    nodes_to_add.append(fused_node)
                    nodes_to_remove.append(node)
                    nodes_to_remove.append(child_node)
                    break

    for node in nodes_to_remove:
        model.graph.node.remove(node)
    model.graph.node.extend(nodes_to_add)

def modify_onnx_model(model_path, modified_onnx_model_path):
    model = onnx.load(model_path)

    for node in model.graph.node:
        if node.op_type == "Add":
            node.op_type = "SystolicAdd"
        elif node.op_type == "BatchNormalization":
            node.op_type = "SystolicBatchNorm"

    fuse_systolic_add_relu(model)
    fuse_systolic_bn_relu(model)

    onnx.save(model, modified_onnx_model_path)
    print(f'Model has been saved as {modified_onnx_model_path}')

if __name__ == "__main__":
    model_path = [] # original onnx model
    modified_onnx_model_path = [] # fused onnx model
    for i in range(len(model_path)):
        modify_onnx_model(model_path[i], modified_onnx_model_path[i])
        print('----------------------------------\n')
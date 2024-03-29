import onnx
from onnx import helper

from add_relu_fuse import fuse_systolic_add_relu
from bn_relu_fuse import fuse_systolic_bn_relu
from delete_description import clear_node_descriptions

model_path = './unet_v2_opt_10k.onnx'
model = onnx.load(model_path)
modified_onnx_model_path = './unet_v2_opt_fused_10k.onnx'

for node in model.graph.node:
    if node.op_type == "BatchNormalization":
        node.op_type = "SystolicBatchNorm"

for node in model.graph.node:
    if node.op_type == "Add":
        node.op_type = "SystolicAdd"
    if node.op_type == "BatchNormalization":
        node.op_type = "SystolicBatchNorm"

fuse_systolic_add_relu(model)
fuse_systolic_bn_relu(model)

onnx.save(model, modified_onnx_model_path)

# clear_node_descriptions(modified_onnx_model_path, modified_onnx_model_path)

print(f'Model has been saved as {modified_onnx_model_path}')

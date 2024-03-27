import onnx


model_path = './unet_v2_opt_fused.onnx' 
model = onnx.load(model_path)

for node in model.graph.node:
    if node.op_type == "BatchNormalization":
        node.op_type = "SystolicBatchNorm"

modified_model_path = './unet_v2_opt_fused_exp.onnx'
onnx.save(model, modified_model_path)

print(f'Model has been modified and saved as {modified_model_path}')
import onnx

def clear_node_descriptions(onnx_model_path, modified_onnx_model_path):

    model = onnx.load(onnx_model_path)

    for node in model.graph.node:
        node.doc_string = ""

    onnx.save(model, modified_onnx_model_path)

onnx_model_path = 'unet_v2_opt_fused_exp.onnx' 
modified_onnx_model_path = 'unet_v2_opt_fused_exp.onnx' 

clear_node_descriptions(onnx_model_path, modified_onnx_model_path)

print(f'Model has been saved as {modified_onnx_model_path}')

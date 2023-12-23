cd /

echo "========================================SpConv3D Starts========================================"

mkdir -p output/

touch /output/trace_spconv3d_gemmini.json
touch /output/trace_spconv3d_cpu.json
touch /output/trace_resnet_gemmini.json
touch /output/trace_resnet_cpu.json
touch /output/trace_unet_cpu.json
touch /output/trace_unet_gemmini.json

# --- SpConv3D ---

# ./ort_test -m spconv3d_v2.onnx -p caffe2 -x 0 -O 99 # single layer-cpu
# mv *.json /output/trace_spconv3d_cpu.json

./ort_test -m spconv3d_v2.onnx -p caffe2 -x 1 -O 99 -t ./trace_spconv3d_gemmini # single layer-gemmini
mv *.json /output/trace_spconv3d_gemmini.json

# --- RESNET ---

# ./ort_test -m resnet_v2.onnx -p caffe2 -x 0 -O 99 # resnet-cpu
# mv *.json /output/trace_resnet_cpu.json

./ort_test -m resnet_v2.onnx -p caffe2 -x 1 -O 99 # resnet-gemmini
mv *.json /output/trace_resnet_gemmini.json

# --- UNET ---

# ./ort_test -m unet_v2.onnx -p caffe2 -x 0 -O 99 -t ./trace_unet_cpu # unet-cpu
# mv *.json /output/trace_unet_cpu.json

./ort_test -m unet_v2.onnx -p caffe2 -x 1 -O 99 -t ./trace_unet_gemmini # unet-gemmini
mv *.json /output/trace_unet_gemmini.json

echo "========================================SpConv3D Ends========================================"

poweroff


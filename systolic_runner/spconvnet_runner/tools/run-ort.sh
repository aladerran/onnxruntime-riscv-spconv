# file under: /home/centos/firesim/target-design/chipyard/generators/gemmini/software/ort_test/run-ort.sh


cd /

echo "========================================SSCN Starts========================================"

mkdir -p output/

# --- UNET ---

rm -f *.csv *.onnx
cp -r data/1k/* .

./ort_test -m unet_v2_1k_fused_opt.onnx -x 2 -O 99 -t ./trace_unet_gemmini # unet-gemmini
mv *.json /output/trace_unet_gemmini.json

rm -f *.csv *.onnx
cp -r data/1k_2/* .

./ort_test -m unet_v2_1k_2_fused.onnx -x 2 -O 99

rm -f *.csv *.onnx
cp -r data/1k_4/* .

./ort_test -m unet_v2_1k_4_fused.onnx -x 2 -O 99

rm -f *.csv *.onnx
cp -r data/1k_8/* .

./ort_test -m unet_v2_1k_8_fused.onnx -x 2 -O 99

rm -f *.csv *.onnx
cp -r data/1k_16/* .

./ort_test -m unet_v2_1k_16_fused.onnx -x 2 -O 99

rm -f *.csv *.onnx
cp -r data/10k/* .

./ort_test -m unet_v2_10k_fused_opt.onnx -x 2 -O 99

rm -f *.csv *.onnx
cp -r data/20k/* .

./ort_test -m unet_v2_20k_fused.onnx -x 2 -O 99

rm -f *.csv *.onnx
cp -r data/50k/* .

./ort_test -m unet_v2_50k_fused.onnx -x 2 -O 99

# --- RESNET ---

./ort_test -m resnet_v2_1k_fused_opt.onnx -x 2 -O 99 -t ./trace_resnet_gemmini # resnet-gemmini
mv *.json /output/trace_resnet_gemmini.json

# mv *.csv /output/

echo "========================================SSCN Ends========================================"


poweroff


#!/bin/bash
set -e

# Change to the project directory
cd ../..

# Build the project in Release mode with parallel compilation
./build.sh --config=Release --parallel

# Change back to the original directory
cd -

./build.sh --config=Release --parallel

# rm -f *.log
rm -f *.csv *.onnx
cp -r data/1k/* .

{
  echo ===================== Runtime begins =====================
  # change the model path to what you want to run/optimize
  spike --extension=gemmini pk ort_test -m unet_v2_1k_fused.onnx -x 2 -O 99 -s ./unet_v2_1k_fused_opt.onnx
  echo ===================== Runtime ends =====================
} >> debug.log 2>&1

mkdir -p ./output
rm -f ./output/*.csv
mv *.csv ./output/
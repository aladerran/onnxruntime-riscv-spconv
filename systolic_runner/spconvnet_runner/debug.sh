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
cp ./tools/*.onnx .

{
  echo ===================== Runtime begins =====================
  # spike --extension=gemmini pk ort_test -m unet_v2_opt_fused.onnx -x 2 -O 99
  spike --extension=gemmini pk ort_test -m unet_v2_opt_fused_exp.onnx -x 2 -O 99
  echo ===================== Runtime ends =====================
} >> debug.log 2>&1

mkdir -p ./output
rm -f ./output/*.csv
mv *.csv ./output/
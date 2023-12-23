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
cp -r data/0.1k/* .

# Run tests and append output to log files
# For each test, replace the placeholder with your ELF file path and other parameters
{
  echo ===================== Runtime begins =====================
  spike --extension=gemmini pk ort_test -m unet_v2.onnx -p caffe2 -x 1 -O 99 -t ./trace_unet_gemmini
  echo ===================== Runtime ends =====================
} >> unet_Gemmini_spike.log 2>&1

{
  echo ===================== Runtime begins =====================
  spike --extension=gemmini pk ort_test -m resnet_v2.onnx -p caffe2 -x 1 -O 99 -t ./trace_resnet_gemmini
  echo ===================== Runtime ends =====================
} >> resnet_Gemmini_spike.log 2>&1

{
  echo ===================== Runtime begins =====================
  spike --extension=gemmini pk ort_test -m unet_v2.onnx -p caffe2 -x 0 -O 99
  echo ===================== Runtime ends =====================
} >> unet_CPU_spike.log 2>&1

{
  echo ===================== Runtime begins =====================
  spike --extension=gemmini pk ort_test -m resnet_v2.onnx -p caffe2 -x 0 -O 99
  echo ===================== Runtime ends =====================
} >> resnet_CPU_spike.log 2>&1


rm -f *.csv *.onnx

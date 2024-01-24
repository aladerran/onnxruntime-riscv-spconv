#!/bin/bash
set -e

# Change to the project directory
cd ../..

# Build the project in Release mode with parallel compilation
./build.sh --config=Release --parallel

# Change back to the original directory
cd -

# Run tests and append output to log files
# For each test, replace the placeholder with your ELF file path and other parameters
{
  echo ===================== Runtime begins =====================
  spike --extension=gemmini pk ort_test -m add.onnx -p caffe2 -x 2 -O 0
  echo ===================== Runtime ends =====================
} >> add_WS_spike.log 2>&1

{
  echo ===================== Runtime begins =====================
  spike --extension=gemmini pk ort_test -m batchnorm.onnx -p caffe2 -x 2 -O 0
  echo ===================== Runtime ends =====================
} >> resnet_WS_spike.log 2>&1


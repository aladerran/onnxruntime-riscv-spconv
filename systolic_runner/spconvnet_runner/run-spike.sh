#!/bin/bash
set -e

./build.sh --config=Release --parallel

{
echo
# Change the following line to your own ELF file
spike --extension=gemmini pk ort_test -m unet_v2.onnx -p caffe2 -x 2 -O 0
} >spike.log 2>&1

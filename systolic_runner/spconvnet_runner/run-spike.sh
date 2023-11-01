#!/bin/bash
set -e

# cd ../..

# ./build.sh --config=Release --parallel

# cd -

# ./build.sh --config=Release --parallel

{
echo
# Change the following line to your own ELF file
spike --extension=gemmini pk ort_test -m unet_v2.onnx -p caffe2 -x 2 -O 0
} >unet_WS_spike.log 2>&1

{
echo
# Change the following line to your own ELF file
spike --extension=gemmini pk ort_test -m resnet_v2.onnx -p caffe2 -x 2 -O 0
} >resnet_WS_spike.log 2>&1

{
echo
# Change the following line to your own ELF file
spike --extension=gemmini pk ort_test -m unet_v2.onnx -p caffe2 -x 0 -O 0
} >unet_CPU_spike.log 2>&1

{
echo
# Change the following line to your own ELF file
spike --extension=gemmini pk ort_test -m resnet_v2.onnx -p caffe2 -x 0 -O 0
} >resnet_CPU_spike.log 2>&1
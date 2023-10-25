#!/bin/bash
set -e

{
cd ~/chipyard/sims/verilator/
echo 
# Change the following line to your own config & ELF file
./simulator-chipyard-CustomGemminiSoCConfig -c pk /ugra/srlin/onnxruntime-riscv-spconv/systolic_runner/spconvnet_runner/ort_test -m /ugra/srlin/onnxruntime-riscv-spconv/systolic_runner/spconvnet_runner/unet_v2.onnx -p caffe2 -x 2 -O 0
echo 
echo ============ cycle accurate info generated ============
} >verilator.log 2>&1

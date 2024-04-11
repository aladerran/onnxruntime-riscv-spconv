#!/bin/bash
set -e

# Change to the ORT directory
cd ../..

# Build the project in Release mode with parallel compilation
./build.sh --config=Release --parallel

# Change back to the original directory
cd -

# ./build.sh --config=Release --parallel
./build.sh --config=Release --parallel --for_firesim

rm -f *.csv *.onnx *.log *.json

cd ~/firesim/sw/firesim-software

./marshal -v --workdir /home/centos/firesim/target-design/chipyard/generators/gemmini/software build ort_test.json
./marshal -v --workdir /home/centos/firesim/target-design/chipyard/generators/gemmini/software install ort_test.json

firesim launchrunfarm && firesim infrasetup && firesim runworkload && firesim terminaterunfarm --forceterminate
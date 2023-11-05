// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <systolic/systolic_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#ifdef FOR_FIRESIM
#include <sys/mman.h>
#endif

#ifdef USE_CUSTOM_OP_LIBRARY
#include "custom_op_library.h"
#endif
#ifdef USE_HWACHA
#include <hwacha/hwacha_provider_factory.h>
#endif

#include "stb_image.h"

#include "tensor_helper.h"
#include "cmd_args.h"


void save_coords(int32_t* data, const int64_t* shape, const char* filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }

    size_t size_0 = shape[0];
    size_t size_1 = shape[1];

    for (size_t i = 0; i < size_0; i++) {
        for (size_t j = 0; j < size_1; j++) {
            outFile << data[i * size_1 + j];
            if (j != size_1 - 1) {
                outFile << ",";
            }
        }
        outFile << std::endl;
    }
    
    outFile.close();
}


void save_feats(float* data, const int64_t* shape, const char* filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }

    size_t size_0 = shape[0];
    size_t size_1 = shape[1];

    for (size_t i = 0; i < size_0; i++) {
        for (size_t j = 0; j < size_1; j++) {
            outFile << data[i * size_1 + j];
            if (j != size_1 - 1) {
                outFile << ",";
            }
        }
        outFile << std::endl;
    }
    
    outFile.close();
}

void save_feats_b(const float* data, const int64_t* shape, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    // if(shape.size()!=2){
    //   std::cerr << "invalid feats shape! get dims:" << shape.size() << std::endl;
    // }

    // 将 shape 写入文件
    int64_t rows = shape[0];
    int64_t cols = shape[1];
    file.write(reinterpret_cast<const char*>(&rows), sizeof(int64_t));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(int64_t));

    // 将数据写入文件
    file.write(reinterpret_cast<const char*>(data), rows * cols * sizeof(float));

    file.close();
}

bool has_suffix(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void read_inputs_coords_feats(const std::string& coords_path, const std::string& feats_path, 
                    std::vector<int32_t> &coords, std::vector<float> &feats){
  std::string line;
  std::stringstream ss;
  std::string value;
  std::ifstream fc(coords_path);
  if (!fc.is_open()) {
    std::cerr << "Failed to open file: " << coords_path << std::endl;
    return;
  }
  size_t count = 0;
  while (std::getline(fc, line)) {
    value.clear();
    ss.clear();
    ss.str(line);
    while (std::getline(ss, value, ',')) {
      coords[count] = static_cast<int32_t>(std::stoi(value));
      count++;
    }
  }
  fc.close();

  std::ifstream ff(feats_path);
  if (!ff.is_open()) {
    std::cerr << "Failed to open file: " << feats_path << std::endl;
    return;
  }
  count = 0;
  while (std::getline(ff, line)) {
    value.clear();
    ss.clear();
    ss.str(line);
    while (std::getline(ss, value, ',')) {
      feats[count] = std::stof(value);
      count++;
    }
  }
  ff.close();
}

unsigned long long read_cycles() {
  unsigned long long cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}


void test_infer(const std::string& preprocess, Ort::Session& session,
                const std::vector<const char*>& input_node_names,
                const std::vector<std::vector<int64_t>>& input_node_shapes,
                const std::vector<size_t>& input_node_sizes,
                const std::vector<const char*>& output_node_names) {
  std::cout << "Processing function test_infer" << std::endl;
  size_t num_inputs = input_node_names.size();
  size_t num_outputs = 6;
  size_t input_coords_idx = 0;
  size_t input_feats_idx = 1;
  size_t input_strides_idx = 2;

  size_t input_coords_size = input_node_sizes[input_coords_idx];
  size_t input_feats_size = input_node_sizes[input_feats_idx];
  size_t input_strides_size = input_node_sizes[input_strides_idx];

  printf("Read test data\n");
  std::vector<int32_t> input_coords_values(input_coords_size);
  std::vector<float> input_feats_values(input_feats_size);
  std::vector<int32_t> input_strides_values(input_strides_size, 1);
  
  read_inputs_coords_feats("coords.csv", "feats.csv", input_coords_values, input_feats_values);

  // create input tensor objects from data values
  std::cout << "Create Tensors" << std::endl;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_coords_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, input_coords_values.data(),
                                                                     input_coords_size, input_node_shapes[input_coords_idx].data(), 2);
  Ort::Value input_feats_tensor = Ort::Value::CreateTensor<float>(memory_info, input_feats_values.data(),
                                                                  input_feats_size, input_node_shapes[input_feats_idx].data(), 2);
  Ort::Value input_strides_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, input_strides_values.data(),
                                                                      input_strides_size, input_node_shapes[input_strides_idx].data(), 1);

  assert(input_coords_tensor.IsTensor());
  assert(input_feats_tensor.IsTensor());
  assert(input_strides_tensor.IsTensor());

  std::vector<Ort::Value> input_tensors;
  // don't assign directly, use move instead to avoid copying a Ort::value
  input_tensors.push_back(std::move(input_coords_tensor));
  input_tensors.push_back(std::move(input_feats_tensor));
  input_tensors.push_back(std::move(input_strides_tensor));

  auto pre_inference_cycles = read_cycles();

  // score model & input tensor, get back output tensor
  std::cout << "Start Running" << std::endl;
  std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                                       input_tensors.data(), num_inputs, output_node_names.data(), num_outputs);
  auto post_inference_cycles = read_cycles();

  assert(output_tensors.size() == num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    assert(output_tensors[i].IsTensor());
  }

  printf("Done! Inference took %llu cycles \n", (post_inference_cycles - pre_inference_cycles));

  // print outputs

  // int32_t* coords_arr = output_tensors[0].GetTensorMutableData<int32_t>();
  // float* feats_arr = output_tensors[1].GetTensorMutableData<float>();
  // int32_t* strides_arr = output_tensors[2].GetTensorMutableData<int32_t>();
  // int32_t* nbmaps_arr = output_tensors[3].GetTensorMutableData<int32_t>();
  // int32_t* nbsizes_arr = output_tensors[4].GetTensorMutableData<int32_t>();
  // int64_t* sizes_io_arr = output_tensors[5].GetTensorMutableData<int64_t>();

  save_coords(coords_arr, output_tensors[0].GetTensorTypeAndShapeInfo().GetShape().data(), "output_coords.csv");
  save_feats(feats_arr, output_tensors[1].GetTensorTypeAndShapeInfo().GetShape().data(), "output_feats.csv");

  return;
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);
  printf("Loaded runner program\n");

  // Use for firesim
#ifdef FOR_FIRESIM
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed");
    exit(1);
  } else {
    printf("Finished mlockall\n");
  }
#endif

  cxxopts::ParseResult cmd = parse(argc, argv);
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  Ort::Env env(static_cast<OrtLoggingLevel>(cmd["debug"].as<int>()), "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  if (cmd.count("trace")) {
    session_options.EnableProfiling(cmd["trace"].as<std::string>().c_str());
  }

  printf("Using systolic in mode %d\n", cmd["execution"].as<int>());
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Systolic(session_options, /*use_arena=*/1, /*accelerator_mode=*/(char)cmd["execution"].as<int>()));
#ifdef USE_HWACHA
  printf("Using hwacha\n");
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Hwacha(session_options, /*use_arena=*/1));
#endif

  // Sets graph optimization level
  // Available levels are
  // 0: ORT_DISABLE_ALL -> To disable all optimizations
  // 1: ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // 2: ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // 99: ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(cmd["optimization"].as<int>()));

  if (cmd.count("save_model")) {
    session_options.SetOptimizedModelFilePath(cmd["save_model"].as<std::string>().c_str());
  }

#ifdef USE_CUSTOM_OP_LIBRARY
  if (cmd.count("kernel")) {
    printf("Loading custom kernel\n");
    Ort::ThrowOnError(RegisterCustomOps((OrtSessionOptions*)session_options, OrtGetApiBase()));
  }
#endif

  // session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);

  const char* model_path = cmd["model"].as<std::string>().c_str();

  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  printf("Create session\n");
  
  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<std::vector<int64_t>> input_node_shapes(num_input_nodes);
  std::vector<size_t> input_node_sizes(num_input_nodes);
  printf("Number of inputs = %zu\n", num_input_nodes);
  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    printf("Input %ld : name=%s, ", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("type=%d, ", type);

    // print input shapes/dims
    input_node_shapes[i] = tensor_info.GetShape();
    input_node_sizes[i] = tensor_info.GetElementCount();
    std::cout << "element count: " << input_node_sizes[i] << std::endl;
    printf("num_dims=%zu: [", input_node_shapes.size());
    for (size_t j = 0; j < input_node_shapes[i].size(); j++) {
      printf("%jd, ", input_node_shapes[i][j]);
    }
    printf("]\n");
  }

  size_t num_output_nodes = session.GetOutputCount();
  printf("Number of outputs = %zu\n", num_output_nodes);
  std::vector<const char*> output_node_names(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; i++) {
    // print output node names
    char* output_name = session.GetOutputName(i, allocator);
    printf("Output %ld : name=%s, ", i, output_name);
    output_node_names[i] = output_name;

    // print output node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("type=%d, ", type);

    // print output shapes/dims
    std::vector<int64_t> output_node_dims = tensor_info.GetShape();
    printf("num_dims=%zu: [", output_node_dims.size());
    for (size_t j = 0; j < output_node_dims.size(); j++) {
      printf("%jd, ", output_node_dims[j]);
    }
    printf("]\n");
  }

  test_infer( cmd["preprocess"].as<std::string>(),
             session, input_node_names, input_node_shapes, input_node_sizes, output_node_names);

  return 0;
}
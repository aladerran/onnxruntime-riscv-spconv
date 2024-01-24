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

#include "cmd_args.h"


void test_infer_matmul(const std::string& preprocess, Ort::Session& session,
                const std::vector<const char*>& input_node_names,
                const std::vector<std::vector<int64_t>>& input_node_shapes,
                const std::vector<size_t>& input_node_sizes,
                const std::vector<const char*>& output_node_names,
                const char* model ) {
  std::cout << "Processing function test_infer mul" << std::endl;
  size_t num_inputs = input_node_names.size();
  size_t num_outputs = 1;

  size_t A_size = 12;
  size_t B_size = 16;
  size_t C_size = 12;

  std::vector<float> A_values(A_size, 0);
  std::vector<float> B_values(B_size, 1);
  for(size_t i = 0; i < A_size; i++){
    A_values[i] = i * 1;
  }

  for(size_t i = 0; i < B_size; i++){
    B_values[i] = 16.0 - i;
  }

  
  std::vector<int64_t> A_shape = {3, 4};
  std::vector<int64_t> B_shape = {4, 4};

  // create input tensor objects from data values
  std::cout << "Create Tensors" << std::endl;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value A_tensor = Ort::Value::CreateTensor<float>(memory_info, A_values.data(), A_size, A_shape.data(), 2);
  Ort::Value B_tensor = Ort::Value::CreateTensor<float>(memory_info, B_values.data(), B_size, B_shape.data(), 2);
  double s = 2.0;

  assert(A_tensor.IsTensor());
  assert(B_tensor.IsTensor());

  std::vector<Ort::Value> input_tensors;
  // don't assign directly, use move instead to avoid copying a Ort::value

  input_tensors.push_back(std::move(A_tensor));
  input_tensors.push_back(std::move(B_tensor));

  // score model & input tensor, get back output tensor
  std::cout << "Start Running" << std::endl;
  std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                                       input_tensors.data(), num_inputs, output_node_names.data(), num_outputs);

  assert(output_tensors.size() == num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    assert(output_tensors[i].IsTensor());
  }

  // print outputs
  float* C_arr = output_tensors[0].GetTensorMutableData<float>();
  for(int i = 0; i < 12; i++){
    printf("%f_", C_arr[i]);
  }
  printf("\nEnd!\n");
  return;
}

void test_infer_add(const std::string& preprocess, Ort::Session& session,
                const std::vector<const char*>& input_node_names,
                const std::vector<std::vector<int64_t>>& input_node_shapes,
                const std::vector<size_t>& input_node_sizes,
                const std::vector<const char*>& output_node_names,
                const char* model ) {
  std::cout << "Processing function test_infer add" << std::endl;
  size_t num_inputs = input_node_names.size();
  size_t num_outputs = 1;

  size_t A_size = 12;
  size_t B_size = 12;
  size_t C_size = 12;

  std::vector<float> A_values(A_size, 0);
  std::vector<float> B_values(B_size, 1);
  for(size_t i = 0; i < A_size; i++){
    A_values[i] = i * 1;
  }

  for(size_t i = 0; i < B_size; i++){
    B_values[i] = 36.0 - i;
  }

  
  std::vector<int64_t> A_shape = {3, 4};
  std::vector<int64_t> B_shape = {3, 4};

  // create input tensor objects from data values
  std::cout << "Create Tensors" << std::endl;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value A_tensor = Ort::Value::CreateTensor<float>(memory_info, A_values.data(), A_size, A_shape.data(), 2);
  Ort::Value B_tensor = Ort::Value::CreateTensor<float>(memory_info, B_values.data(), B_size, B_shape.data(), 2);
  double s = 2.0;

  assert(A_tensor.IsTensor());
  assert(B_tensor.IsTensor());

  std::vector<Ort::Value> input_tensors;
  // don't assign directly, use move instead to avoid copying a Ort::value

  input_tensors.push_back(std::move(A_tensor));
  input_tensors.push_back(std::move(B_tensor));

  // score model & input tensor, get back output tensor
  std::cout << "Start Running" << std::endl;
  std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                                       input_tensors.data(), num_inputs, output_node_names.data(), num_outputs);

  assert(output_tensors.size() == num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    assert(output_tensors[i].IsTensor());
  }

  // print outputs
  float* C_arr = output_tensors[0].GetTensorMutableData<float>();
  for(int i = 0; i < 12; i++){
    printf("%f_", C_arr[i]);
  }
  printf("\nEnd!\n");
  return;
}

void test_infer_batchnrom(const std::string& preprocess, Ort::Session& session,
                const std::vector<const char*>& input_node_names,
                const std::vector<std::vector<int64_t>>& input_node_shapes,
                const std::vector<size_t>& input_node_sizes,
                const std::vector<const char*>& output_node_names,
                const char* model ) {
  std::cout << "Processing function test_infer batchnorm" << std::endl;
  size_t num_inputs = input_node_names.size();
  size_t num_outputs = 1;

  size_t A_size = 24;


  std::vector<float> A_values(A_size, 0);
  for(int i = 0; i < A_size; i++){
    A_values[i] = i * 1.0;
  }
  
  std::vector<int64_t> A_shape = {4, 3, 2};

  // create input tensor objects from data values
  std::cout << "Create Tensors" << std::endl;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value A_tensor = Ort::Value::CreateTensor<float>(memory_info, A_values.data(), A_size, A_shape.data(), 3);

  assert(A_tensor.IsTensor());

  std::vector<Ort::Value> input_tensors;
  // don't assign directly, use move instead to avoid copying a Ort::value

  input_tensors.push_back(std::move(A_tensor));

  // score model & input tensor, get back output tensor
  std::cout << "Start Running" << std::endl;
  std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                                       input_tensors.data(), num_inputs, output_node_names.data(), num_outputs);

  assert(output_tensors.size() == num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    assert(output_tensors[i].IsTensor());
  }

  // print outputs
  float* C_arr = output_tensors[0].GetTensorMutableData<float>();
  for(int i = 0; i < 4; i++){
    printf("[");
    for (int j = 0; j < 3; j++){
      printf("[%f %f]\n", C_arr[i*6+j*2], C_arr[i*6+j*2+1]);
    }
    printf("]\n");
  }
  printf("\nEnd!\n");
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
    printf("num_dims=%zu: [", input_node_shapes[i].size());
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

  if(strcmp(model_path,"add.onnx") == 0){
    test_infer_add( cmd["preprocess"].as<std::string>(),
             session, input_node_names, input_node_shapes, input_node_sizes, output_node_names, model_path);
  } else if(strcmp(model_path, "batchnorm.onnx") == 0 ){
    test_infer_batchnrom( cmd["preprocess"].as<std::string>(),
             session, input_node_names, input_node_shapes, input_node_sizes, output_node_names, model_path);
  } else if (strcmp(model_path, "mul.onnx") == 0) {
    test_infer_matmul( cmd["preprocess"].as<std::string>(),
             session, input_node_names, input_node_shapes, input_node_sizes, output_node_names, model_path);
  }
  

  return 0;
}
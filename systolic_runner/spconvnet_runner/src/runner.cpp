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


bool has_suffix(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int getLabelOfBatchImage(const std::string& path) {
  size_t lastidx = path.find_last_of("/\\");
  size_t secondlastidx = path.find_last_of("/\\", lastidx - 1);
  return std::stoi(path.substr(secondlastidx + 1, (lastidx - secondlastidx - 1)));
}


unsigned long long read_cycles() {
  unsigned long long cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}


void print_tensor_dim2(const char* name, int32_t* data, const int64_t* shape) {
  size_t size_0 = shape[0];
  size_t size_1 = shape[1];
  std::cout << name << " [" << std::endl;
  for (size_t i = 0; i < size_0; i++) {
    std::cout <<" [" ;
    for (size_t j = 0; j < size_1; j++) {
      std::cout << data[i * size_1 + j] << " ";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "]" << std::endl;
}
// template<typename T>
void print_tensor_dim2(const char* name, int64_t* data, const int64_t* shape) {
  size_t size_0 = shape[0];
  size_t size_1 = shape[1];
  std::cout << name << " [" << std::endl;
  for (size_t i = 0; i < size_0; i++) {
    std::cout <<" [" ;
    for (size_t j = 0; j < size_1; j++) {
      std::cout << data[i * size_1 + j] << " ";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "]" << std::endl;
}
void print_tensor_dim2(const char* name, float* data, const int64_t* shape) {
  size_t size_0 = shape[0];
  size_t size_1 = shape[1];
  std::cout << name << " [" << std::endl;
  for (size_t i = 0; i < size_0; i++) {
    std::cout <<" [" ;
    for (size_t j = 0; j < size_1; j++) {
      std::cout << data[i * size_1 + j] << " ";
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "]" << std::endl;
}
// template<typename T>
void print_tensor_dim3(const char* name, float* data, const int64_t* shape) {
  size_t size_0 = shape[0];
  size_t size_1 = shape[1];
  size_t size_2 = shape[2];
  std::cout << name << " [" << std::endl;
  for (size_t i = 0; i < size_0; i++) {
    std::cout << " [";
    for (size_t j = 0; j < size_1; j++) {
      std::cout << " [";
      for (size_t k = 0; k < size_2; k++) {
        std::cout << data[(i * size_1 + j) * size_2 + k] << " ";
      }
      std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "]" << std::endl;
}
void print_tensor_dim3(const char* name, float* data, const int32_t* shape) {
  size_t size_0 = shape[0];
  size_t size_1 = shape[1];
  size_t size_2 = shape[2];
  std::cout << name << " [" << std::endl;
  for (size_t i = 0; i < size_0; i++) {
    std::cout << " [";
    for (size_t j = 0; j < size_1; j++) {
      std::cout << " [";
      for (size_t k = 0; k < size_2; k++) {
        std::cout << data[(i * size_1 + j) * size_2 + k] << " ";
      }
      std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "]" << std::endl;
}

void test_infer(const std::string& preprocess, Ort::Session& session,
                const std::vector<const char*>& input_node_names,
                const std::vector<std::vector<int64_t>>& input_node_shapes,
                const std::vector<size_t>& input_node_sizes,
                const std::vector<const char*>& output_node_names) {
  std::cout << "Processing function test_infer" << std::endl;
  size_t num_inputs = input_node_names.size();
  size_t num_outputs = 6;
  // size_t input_tensor_size = 20 * 4;  // simplify ... using known dim values to calculate size
  //                                           // use OrtGetTensorShapeElementCount() to get official size!
  size_t input_coords_idx = 0;
  size_t input_feats_idx = 1;
  size_t input_strides_idx = 2;
  size_t input_weight_idx = 3;

  size_t input_coords_size = input_node_sizes[input_coords_idx];
  size_t input_feats_size = input_node_sizes[input_feats_idx];
  size_t input_strides_size = input_node_sizes[input_strides_idx];
  size_t input_weight_size = input_node_sizes[input_weight_idx];

  printf("Generate test data\n");
  std::vector<int64_t> input_coords_values(input_coords_size);
  std::vector<float> input_feats_values(input_feats_size);
  std::vector<int64_t> input_strides_values(input_strides_size, 1);
  std::vector<float> input_weight_values(input_weight_size);

  for (unsigned int i = 0; i < input_feats_size; i++)
    input_feats_values[i] = (float)i / (input_feats_size + 1);

  for (unsigned int i = 0; i < 108U; i++)
    input_weight_values[i] = (float)i / 109U;

  for (unsigned int i = 0; i < input_coords_size; i++)
    input_coords_values[i] = static_cast<int64_t>(i);

  // create input tensor objects from data values
  std::cout << "Create Tensors" << std::endl;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_coords_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_coords_values.data(),
                                                                     input_coords_size, input_node_shapes[input_coords_idx].data(), 2);
  Ort::Value input_feats_tensor = Ort::Value::CreateTensor<float>(memory_info, input_feats_values.data(),
                                                                  input_feats_size, input_node_shapes[input_feats_idx].data(), 2);
  Ort::Value input_strides_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_strides_values.data(),
                                                                      input_strides_size, input_node_shapes[input_strides_idx].data(), 1);
  Ort::Value input_weight_tensor = Ort::Value::CreateTensor<float>(memory_info, input_weight_values.data(),
                                                                   input_weight_size, input_node_shapes[input_weight_idx].data(), 3);
  assert(input_coords_tensor.IsTensor());
  assert(input_feats_tensor.IsTensor());
  assert(input_strides_tensor.IsTensor());
  assert(input_weight_tensor.IsTensor());

  std::vector<Ort::Value> input_tensors;
  // don't assign directly, use move instead to avoid copying a Ort::value
  input_tensors.push_back(std::move(input_coords_tensor));
  input_tensors.push_back(std::move(input_feats_tensor));
  input_tensors.push_back(std::move(input_strides_tensor));
  input_tensors.push_back(std::move(input_weight_tensor));

  // auto pre_inference_cycles = read_cycles();

  // score model & input tensor, get back output tensor
  // auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), 4, output_node_names.data(), 6);
  std::cout << "Start Running" << std::endl;
  std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                                       input_tensors.data(), num_inputs, output_node_names.data(), num_outputs);
  // auto post_inference_cycles = read_cycles();

  assert(output_tensors.size() == num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    assert(output_tensors[i].IsTensor());
  }

  // print outputs

  int64_t* coords_arr = output_tensors[0].GetTensorMutableData<int64_t>();
  float* feats_arr = output_tensors[1].GetTensorMutableData<float>();
  int64_t* strides_arr = output_tensors[2].GetTensorMutableData<int64_t>();
  int64_t* nbmaps_arr = output_tensors[3].GetTensorMutableData<int64_t>();
  int64_t* nbsizes_arr = output_tensors[4].GetTensorMutableData<int64_t>();
  //int64_t* sizes_io_arr = output_tensors[5].GetTensorMutableData<int64_t>();

  print_tensor_dim2(output_node_names[0], coords_arr, output_tensors[0].GetTensorTypeAndShapeInfo().GetShape().data());
  print_tensor_dim2(output_node_names[1], feats_arr, output_tensors[1].GetTensorTypeAndShapeInfo().GetShape().data());
  print_tensor_dim2(output_node_names[2], strides_arr, output_tensors[2].GetTensorTypeAndShapeInfo().GetShape().data());
  print_tensor_dim2(output_node_names[3], nbmaps_arr, output_tensors[3].GetTensorTypeAndShapeInfo().GetShape().data());
  print_tensor_dim2(output_node_names[4], nbsizes_arr, output_tensors[4].GetTensorTypeAndShapeInfo().GetShape().data());


  return;
}

// #include <sys/mman.h>

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

  session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);

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
    printf("num_dims=%zu: [", input_node_shapes.size());
    for (size_t j = 0; j < input_node_shapes[i].size(); j++) {
      printf("%jd, ", input_node_shapes[i][j]);
    }
    printf("]\n");
  }

  // if (num_input_nodes > 1) {
  //   printf("ERROR: Graph has multiple input nodes defined.\n");
  //   return -1;
  // }

  // Results should be...
  // Number of inputs = 1
  // Input 0 : name = data_0
  // Input 0 : type = 1
  // Input 0 : num_dims = 4
  // Input 0 : dim 0 = 1
  // Input 0 : dim 1 = 3
  // Input 0 : dim 2 = 224
  // Input 0 : dim 3 = 224

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

  // if (output_node_names.size() > 1) {
  //   printf("ERROR: Graph has multiple output nodes defined. Please specify an output manually.\n");
  //   return -1;
  // }

  test_infer( cmd["preprocess"].as<std::string>(),
             session, input_node_names, input_node_shapes, input_node_sizes, output_node_names);

  return 0;
}

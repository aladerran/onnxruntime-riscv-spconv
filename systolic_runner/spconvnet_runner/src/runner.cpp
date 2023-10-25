// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <vector>
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
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


// Functions to print out point-cloud data
void print_inputData(const char* name, int* feat, int nrows, int cols) {
    std::cout << "=====" << name << "=====" << std::endl;
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << feat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// Functions to generate point-cloud data
struct Point {
    int x, y, z;
};

struct HashFunction {
    std::size_t operator()(const Point& p) const {
        return std::hash<int>()(p.x) ^ std::hash<int>()(p.y) ^ std::hash<int>()(p.z);
    }
};

bool operator==(const Point& p1, const Point& p2) {
    return p1.x == p2.x && p1.y == p2.y && p1.z == p2.z;
}

std::vector<Point> sparseQuantize(const std::vector<Point>& coords, float voxelSize) {
    std::unordered_set<Point, HashFunction> uniqueCoords;
    for (const auto& coord : coords) {
        Point quantizedCoord{
            static_cast<int>(std::floor(coord.x / voxelSize)),
            static_cast<int>(std::floor(coord.y / voxelSize)),
            static_cast<int>(std::floor(coord.z / voxelSize))
        };
        uniqueCoords.insert(quantizedCoord);
    }

    std::vector<Point> result(uniqueCoords.begin(), uniqueCoords.end());
    return result;
}

std::vector<int> append_batchInfo(const std::vector<int>& originalCoords) {
    std::vector<int> newCoords(originalCoords.size() / 3 * 4, 0);

    for (size_t i = 0; i < originalCoords.size() / 3; ++i) {
        newCoords[i * 4] = originalCoords[i * 3];        
        newCoords[i * 4 + 1] = originalCoords[i * 3 + 1]; 
        newCoords[i * 4 + 2] = originalCoords[i * 3 + 2]; 
        // Fourth dimension is already initialized to 0
    }

    return newCoords;
}

std::pair<std::vector<int>, std::vector<float>> generateRandomPointCloud(int size = 10000, float voxelSize = 0.2) {
    std::vector<int> coords(size * 3);  // Flatten to 1D vector
    std::vector<float> feats(size * 4);  // Flatten to 1D vector

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < 3; ++j) {
            coords[i * 3 + j] = static_cast<int>(rand() % 100);  // Random coordinates
            feats[i * 4 + j] = static_cast<float>(rand()) / RAND_MAX * 10.0f;  // Random features
        }
        feats[i * 4 + 3] = static_cast<float>(rand()) / RAND_MAX * 10.0f;  // Random features
    }

    std::vector<Point> pointCoords(size);
    for (int i = 0; i < size; ++i) {
        pointCoords[i] = {coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]};
    }

    std::vector<Point> uniqueCoords = sparseQuantize(pointCoords, voxelSize);

    std::vector<int> resultCoords(uniqueCoords.size() * 3);
    for (size_t i = 0; i < uniqueCoords.size(); ++i) {
        resultCoords[i * 3] = uniqueCoords[i].x;
        resultCoords[i * 3 + 1] = uniqueCoords[i].y;
        resultCoords[i * 3 + 2] = uniqueCoords[i].z;
    }

    return {append_batchInfo(resultCoords), feats};
}

void save_input_data(const std::vector<int32_t>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i];
        if (i != data.size() - 1) {
            file << ",";
        }
    }
    file.close();
}

void save_input_data(const std::vector<float>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i];
        if (i != data.size() - 1) {
            file << ",";
        }
    }
    file.close();
}


void generateRandomValues(std::vector<float>& input) {
    std::random_device rd; 
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> dis(-1000, 1000);

    for (unsigned int i = 0; i < input.size(); i++) {
        input[i] = dis(gen); 
    }

}

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


void print_tensor_dim1(const char* name, int32_t* data, const int64_t* shape) {
  size_t size_0 = shape[0];
  std::cout << name << " [" << std::endl;
  for (size_t i = 0; i < size_0; i++) {
    std::cout << data[i] << " ";
  }
  std::cout << "]" << std::endl;
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
  size_t input_coords_idx = 0;
  size_t input_feats_idx = 1;
  size_t input_strides_idx = 2;

  size_t input_coords_size = input_node_sizes[input_coords_idx];
  size_t input_feats_size = input_node_sizes[input_feats_idx];
  size_t input_strides_size = input_node_sizes[input_strides_idx];

  std::cout << "Generating random inputs..." << std::endl;

  // generate random input data
  auto PC = generateRandomPointCloud(input_coords_size/4);
  std::vector<int32_t> input_coords_values = std::get<0>(PC);
  std::vector<float> input_feats_values(input_feats_size);
  generateRandomValues(input_feats_values);
  std::vector<int32_t> input_strides_values(input_strides_size, 1);
  save_input_data(input_feats_values, "input_feats.csv");
  save_input_data(input_coords_values, "input_coords.csv");

  // create input tensor objects from data values
  std::cout << "Creating tensors..." << std::endl;
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

  auto start_time = std::chrono::high_resolution_clock::now();
  auto pre_inference_cycles = read_cycles();

  // score model & input tensor, get back output tensor
  // auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(), 4, output_node_names.data(), 6);
  std::cout << "Start Running" << std::endl;
  std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                                       input_tensors.data(), num_inputs, output_node_names.data(), num_outputs);
  auto post_inference_cycles = read_cycles();
  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  assert(output_tensors.size() == num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    assert(output_tensors[i].IsTensor());
  }

  printf("==========================Results==========================\n");

  printf("Done! Inference took %llu cycles \n", (post_inference_cycles - pre_inference_cycles));
  std::cout << "Execution Time: " << elapsed_time.count() << " milliseconds" << std::endl;

  // print input & outputs

  int32_t* coords_arr = output_tensors[0].GetTensorMutableData<int32_t>();
  // int32_t* strides_arr = output_tensors[2].GetTensorMutableData<int32_t>();
  // int32_t* nbmaps_arr = output_tensors[3].GetTensorMutableData<int32_t>();
  // int32_t* nbsizes_arr = output_tensors[4].GetTensorMutableData<int32_t>();
  float* feats_arr = output_tensors[1].GetTensorMutableData<float>();

  std::cout << "output_coords:" << std::endl;
  print_tensor_dim2(output_node_names[0], coords_arr, output_tensors[0].GetTensorTypeAndShapeInfo().GetShape().data());
  std::cout << "output_feats:" << std::endl;
  print_tensor_dim2(output_node_names[1], feats_arr, output_tensors[1].GetTensorTypeAndShapeInfo().GetShape().data());
  // std::cout << "output_strides:" << std::endl;
  // print_tensor_dim1(output_node_names[2], strides_arr, output_tensors[2].GetTensorTypeAndShapeInfo().GetShape().data());
  // std::cout << "output_nbmaps:" << std::endl;
  // print_tensor_dim2(output_node_names[3], nbmaps_arr, output_tensors[3].GetTensorTypeAndShapeInfo().GetShape().data());
  // std::cout << "output_nbsizes:" << std::endl;
  // print_tensor_dim1(output_node_names[4], nbsizes_arr, output_tensors[4].GetTensorTypeAndShapeInfo().GetShape().data());

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

#include "spconv.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/safeint.h"
#include "conv_pool_helper.h"
#include <algorithm> // std::sort needs this
#include <cstdint>

// #ifdef SYSTOLIC_FP32

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_KERNEL_EX(
    SpConv3d,
    kOnnxDomain,
    14,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SpConv3d<float>);

unsigned long long map_cycles;
unsigned long long conv_cycles;
unsigned long long total_cycles;
unsigned long long downsample_cycles;
unsigned long long kernel_offset_cycles;
unsigned long long parse_nbsizes_cycles;
unsigned long long parse_nbmaps_cycles;

template <typename T>
Status SpConv3d<T>::Compute(OpKernelContext* context) const {

  auto total_start = read_cycles();

  const Tensor* InputCoords = context->Input<Tensor>(0);
  const Tensor* InputFeats = context->Input<Tensor>(1);
  const Tensor* InputStrides = context->Input<Tensor>(2);
  const Tensor* Weight = context->Input<Tensor>(3);
  const Tensor* Nbmaps_i = context->Input<Tensor>(4);         // optional. nullptr if not provided
  const Tensor* Nbsizes_i = context->Input<Tensor>(5);        // optional. nullptr if not provided
  const Tensor* SizesIO_i = context->Input<Tensor>(6);        // optional. nullptr if not provided
  const Tensor* OutputCoords_i = context->Input<Tensor>(7);   // optional. nullptr if not provided
  const Tensor* Bias = context->Input<Tensor>(8);             // optional. nullptr if not provided

    // OutputCoords_i, Nbmaps_i, Nbsizes_i, SizesIO_i should be provided togather, or none of them is provided

  // std::cout << "debug-zxr: start spconv3d computing" << std::endl;
  std::cout << "debug-zxr:domain:" << context->GetOpDomain() << " /type:" << context->GetOpType() << " /name:" << context->GetNodeName() << std::endl;
  std::string node_name = context->GetNodeName();


  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(InputCoords, InputFeats, InputStrides, Weight));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(Weight->Shape(), kernel_shape));

  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }
  const bool transposed = conv_attrs_.transposed == 1; 
  
  Tensor* OutputCoords;
  Tensor* OutputFeats;
  Tensor* OutputStrides = context->Output(2, InputStrides->Shape());
  Tensor* Nbmaps_o;
  Tensor* Nbsizes_o;
  std::vector<int64_t> sizes_io_shape({2});
  Tensor* SizesIO_o = context->Output(5, sizes_io_shape); 

  const int* input_coords_data = InputCoords->template Data<int32_t>();
  const float* input_feats_data = InputFeats->template Data<float>();
  const int* input_strides_data = InputStrides->template Data<int32_t>();
  const float* weight_data = Weight->template Data<float>();

  if ( !transposed ){
    // std::cout << "debug-zxr: start buildkmap" << std::endl;
    ORT_RETURN_IF_ERROR(BuildKmap(context, InputCoords, InputStrides, OutputCoords, Nbmaps_o, Nbsizes_o));
    int32_t* sizes_io_data = SizesIO_o -> MutableData<int32_t>();
    sizes_io_data[0] = static_cast<int32_t>(InputCoords->Shape()[0]);
    sizes_io_data[1] = static_cast<int32_t>(OutputCoords->Shape()[0]);
    std::vector<int64_t> output_feats_shape({OutputCoords->Shape()[0], Weight->Shape()[2]});
    OutputFeats = context->Output(1, output_feats_shape);

    ORT_RETURN_IF_ERROR(ConvolutionForward(InputFeats, OutputFeats, Weight, Nbmaps_o, Nbsizes_o));
    int* output_strides_data = OutputStrides-> template MutableData<int32_t>();
    for (size_t i = 0; i < 3; i++){
      output_strides_data[i] = input_strides_data[i] * strides[i];
    }
  }else {
    OutputCoords = context->Output(0, OutputCoords_i->Shape());
    Nbmaps_o = context->Output(3, Nbmaps_i->Shape());
    Nbsizes_o = context->Output(4, Nbsizes_i->Shape());
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(OutputCoords_i, OutputCoords));
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(Nbmaps_i, Nbmaps_o));
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(Nbsizes_i, Nbsizes_o));
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(SizesIO_i, SizesIO_o));

    std::vector<int64_t> output_feats_shape({OutputCoords->Shape()[0], Weight->Shape()[2]});
    OutputFeats = context->Output(1, output_feats_shape);
    ORT_RETURN_IF_ERROR(ConvolutionForward(InputFeats, OutputFeats, Weight, Nbmaps_o, Nbsizes_o));
    int32_t* output_strides_data = OutputStrides -> template MutableData<int32_t>();
    for (size_t i = 0; i < 3; i++){
      output_strides_data[i] = input_strides_data[i] / strides[i];
    }
  }

  total_cycles += read_cycles() - total_start;

  if(context->GetNodeName() == "/fuse/resblock2/main/conv3d2_3/SpConv3d"
    || context->GetNodeName() == "/4/4.0/conv3d/SpConv3d"){
    auto print_start = read_cycles();
    std::cout << "==================" << std::endl;
    std::cout << "Spconv3D Profiling:" << std::endl;
    print_cycles_backend();
    std::cout << "==================" << std::endl;
    std::cout << "Compute kernel offset cycles: " << kernel_offset_cycles << std::endl;
    std::cout << "Downsample cycles: " << downsample_cycles << std::endl;
    std::cout << "Parse nbsizes cycles: " << parse_nbsizes_cycles << std::endl;
    std::cout << "Parse nbmaps cycles: " << parse_nbmaps_cycles << std::endl;
    std::cout << "Map processing cycles: " << map_cycles << std::endl;
    std::cout << "Convolution forward cycles: " << conv_cycles << std::endl;
    std::cout << "SpConv3D total cycles: " << total_cycles << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Printf cycles: " << read_cycles() - print_start << std::endl;
  }

  return Status::OK();
}

bool cmp(std::vector<int32_t> &a, std::vector<int32_t> &b) {
  if(a[3] != b[3]){
    return a[3] < b[3];
  } else if(a[0] != b[0]){
    return a[0] < b[0];
  } else if(a[1] != b[1]){
    return a[1] < b[1];
  } else {
    return a[2] < b[2];
  }
}


template <typename T>
Status SpConv3d<T>::BuildKmap(OpKernelContext * context, const Tensor* InputCoords, const Tensor* InputStrides, 
                              Tensor* &OutputCoords, Tensor* &Nbmaps, Tensor* &Nbsizes) const {

  unsigned long long map_start = read_cycles();

  std::vector<int64_t> kernel_shape = conv_attrs_.kernel_shape_;
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }
  const int* input_strides_data = InputStrides->template Data<int32_t>();
  // build kernel offsets
  auto kernel_offset_start = read_cycles();
  std::vector<int32_t> x_dim_offsets(kernel_shape[0]);
  std::vector<int32_t> y_dim_offsets(kernel_shape[1]);
  std::vector<int32_t> z_dim_offsets(kernel_shape[2]);
  for(size_t i = 0; i < kernel_shape[0]; i++){
    x_dim_offsets[i] = i - (kernel_shape[0] - 1)/2 ;
  }
  for(size_t i = 0; i < kernel_shape[1]; i++){
    y_dim_offsets[i] = i - (kernel_shape[1] - 1)/2;
  }
  for(size_t i =0; i < kernel_shape[2]; i++){
    z_dim_offsets[i] = i - (kernel_shape[2] - 1)/2;
  }
  int32_t kernel_volume = static_cast<int32_t>(kernel_shape[0] * kernel_shape[1] * kernel_shape[2]);
  std::vector<int32_t> offsets(kernel_volume * 3);
  if(kernel_volume % 2 == 1){
    for (size_t i = 0; i < kernel_shape[2]; i++) {
      for (size_t j = 0; j < kernel_shape[1]; j++) {
        for (size_t k = 0; k < kernel_shape[0]; k++) {
          offsets[( i * kernel_shape[0] * kernel_shape[1] + j * kernel_shape[0] + k ) * 3] = x_dim_offsets[k] * input_strides_data[0];
          offsets[( i * kernel_shape[0] * kernel_shape[1] + j * kernel_shape[0] + k ) * 3 + 1] = y_dim_offsets[j] * input_strides_data[1];
          offsets[( i * kernel_shape[0] * kernel_shape[1] + j * kernel_shape[0] + k ) * 3 + 2] = z_dim_offsets[i] * input_strides_data[2];
        }
      }
    }
  } else {
    for (size_t i = 0; i < kernel_shape[0]; i++) {
      for (size_t j = 0; j < kernel_shape[1]; j++) {
        for (size_t k = 0; k < kernel_shape[2]; k++) {
          offsets[( i * kernel_shape[1] * kernel_shape[2] + j * kernel_shape[2] + k ) * 3] = x_dim_offsets[i] * input_strides_data[0];
          offsets[( i * kernel_shape[1] * kernel_shape[2] + j * kernel_shape[2] + k ) * 3 + 1] = y_dim_offsets[j] * input_strides_data[1];
          offsets[( i * kernel_shape[1] * kernel_shape[2] + j * kernel_shape[2] + k ) * 3 + 2] = z_dim_offsets[k] * input_strides_data[2];
        }
      }
    }
  }
  
  kernel_offset_cycles += read_cycles() - kernel_offset_start;
  
// ==========================================================================================

  const TensorShape& input_coords_shape = InputCoords -> Shape();
  const int* input_coords_data = InputCoords-> template Data<int32_t>();
  std::vector<uint32_t> references(input_coords_shape[0]);
  hash_cpu_uint32t(input_coords_data, references.data(), input_coords_shape[0]);

  int* output_coords_data;
  int64_t num_output_coords;
  size_t count;
  auto downsample_start = read_cycles();

  if ( conv_attrs_.strides[0] > 1 || conv_attrs_.strides[1] > 1 || conv_attrs_.strides[2] > 1) {
    //downsampling
    std::vector<std::vector<int32_t>> output_coords_vector_uncoalesced;
    if ((strides[0] == 1 || strides[0] == kernel_shape[0]) 
          && (strides[1] == 1 || strides[1] == kernel_shape[1]) 
          && (strides[2] == 1 || strides[2] == kernel_shape[2])){
      std::vector<int32_t> sample_stride(3); 
      for (size_t i = 0; i < 3; i++) {
        sample_stride[i] = strides[i] * input_strides_data[i];
      }

      output_coords_vector_uncoalesced.resize(input_coords_shape[0], std::vector<int32_t>(4));

      for (size_t i = 0; i < input_coords_shape[0]; i++) {
        output_coords_vector_uncoalesced[i][0] = input_coords_data[ i * 4 ] / sample_stride[0] * sample_stride[0];
        output_coords_vector_uncoalesced[i][1] = input_coords_data[ i * 4 + 1 ] / sample_stride[1] * sample_stride[1];
        output_coords_vector_uncoalesced[i][2] = input_coords_data[ i * 4 + 2 ] / sample_stride[2] * sample_stride[2];
        output_coords_vector_uncoalesced[i][3] = input_coords_data[ i * 4 + 3 ];
      }
      
    } else {
      std::vector<int32_t> sample_stride(3); 
      for (size_t i = 0; i < 3; i++) {
        sample_stride[i] = strides[i] * input_strides_data[i];
      }
      int32_t min[3] = {INT32_MAX, INT32_MAX, INT32_MAX};
      int32_t max[3] = {INT32_MIN, INT32_MIN, INT32_MIN};

      for(size_t i = 0; i < input_coords_shape[0]; i++) {
        min[0] = min[0] <= input_coords_data[i * 4] ? min[0] : input_coords_data[i * 4];
        max[0] = max[0] >= input_coords_data[i * 4] ? max[0] : input_coords_data[i * 4];
        min[1] = min[1] <= input_coords_data[i * 4 + 1] ? min[1] : input_coords_data[i * 4 + 1];
        max[1] = max[1] >= input_coords_data[i * 4 + 1] ? max[1] : input_coords_data[i * 4 + 1];
        min[2] = min[2] <= input_coords_data[i * 4 + 2] ? min[2] : input_coords_data[i * 4 + 2];
        max[2] = max[2] >= input_coords_data[i * 4 + 2] ? max[2] : input_coords_data[i * 4 + 2];
      }

      int uncoalesced_size = input_coords_shape[0] * ( kernel_shape[0] / strides[0] + 1) * \
                                                     ( kernel_shape[1] / strides[1] + 1) * \
                                                     ( kernel_shape[2] / strides[2] + 1);
      output_coords_vector_uncoalesced.resize(uncoalesced_size, std::vector<int32_t>(4));
      count = 0;
      for(size_t i = 0; i < input_coords_shape[0]; i++) {
        for(size_t j = 0; j < kernel_volume; j++) {
          int coord[4];
          coord[0] = input_coords_data[ i * 4 ] + offsets[ j * 3];
          coord[1] = input_coords_data[ i * 4 + 1] + offsets[ j * 3 + 1];
          coord[2] = input_coords_data[ i * 4 + 2] + offsets[ j * 3 + 2];
          coord[3] = input_coords_data[ i * 4 + 3];
          if( coord[0] % sample_stride[0] == 0 && coord[0] >= min[0] /* &&coord[0] <= max[0] */ 
              && coord[1] % sample_stride[1] == 0 && coord[1] >= min[1] /* &&coord[0] <= max[0] */ 
              && coord[2] % sample_stride[2] == 0 && coord[2] >= min[2] /* &&coord[0] <= max[0] */ ){
            output_coords_vector_uncoalesced[count][0] = coord[0];
            output_coords_vector_uncoalesced[count][1] = coord[1];
            output_coords_vector_uncoalesced[count][2] = coord[2];
            output_coords_vector_uncoalesced[count][3] = coord[3];
            count++;
          }
        }
      }
      output_coords_vector_uncoalesced.resize(count);      
    }

    std::sort(output_coords_vector_uncoalesced.begin(), output_coords_vector_uncoalesced.end(), cmp);
    auto pos = std::unique(output_coords_vector_uncoalesced.begin(), output_coords_vector_uncoalesced.end());
    output_coords_vector_uncoalesced.erase(pos, output_coords_vector_uncoalesced.end());

    num_output_coords = output_coords_vector_uncoalesced.size();
    std::vector<int64_t> output_coords_shape({num_output_coords, 4});
    OutputCoords = context->Output(0, output_coords_shape);
    output_coords_data = OutputCoords -> template MutableData<int32_t>();
    for (size_t i = 0; i < num_output_coords; ++i){
      output_coords_data[i * 4] = output_coords_vector_uncoalesced[i][0];
      output_coords_data[i * 4 + 1] = output_coords_vector_uncoalesced[i][1];
      output_coords_data[i * 4 + 2] = output_coords_vector_uncoalesced[i][2];
      output_coords_data[i * 4 + 3] = output_coords_vector_uncoalesced[i][3];
    }
  } else {
    // use input coordinates as output coordinates
    OutputCoords = context->Output(0, InputCoords->Shape());
    num_output_coords = InputCoords->Shape()[0];
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(InputCoords, OutputCoords));
    output_coords_data = OutputCoords -> template MutableData<int32_t>();
  }
  downsample_cycles += read_cycles() - downsample_start;

// ==========================================================================================

  std::vector<uint32_t> queries(num_output_coords * kernel_volume);
  kernel_hash_cpu_uint32t(output_coords_data, offsets.data(), queries.data(), num_output_coords, kernel_volume);

  std::vector<uint32_t> indices(references.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::vector<uint32_t> results(queries.size());
  hash_query_cpu_uint32t(queries.data(), references.data(), indices.data(), results.data(), references.size(), queries.size());

// ==========================================================================================

  //parse nbsizes
  std::vector<int64_t> nbsizes_shape({kernel_volume});
  Nbsizes = context->Output(4, nbsizes_shape);
  // std::cout << "debug-zxr: parse nbsizes" << std::endl;
  auto parse_nbsizes_start = read_cycles();
  int* nbsizes_data = Nbsizes->template MutableData<int32_t>();
  for (size_t i = 0; i < kernel_volume; i++){
    size_t sum = 0;
    for(size_t j = 0; j < num_output_coords; j++){
      int result = results[i * num_output_coords + j];
      if(results[i * num_output_coords + j] != 0) {
        sum++;
      }
    }
    nbsizes_data[i] = sum;
  }

  parse_nbsizes_cycles += read_cycles() - parse_nbsizes_start;

  //parse nbmaps
  // std::cout << "debug-zxr: parse nbmaps" << std::endl;
  auto parse_nbmaps_start = read_cycles();
  int64_t sum_nbmaps = 0;
  for(size_t i = 0; i < kernel_volume; i++){
    sum_nbmaps += nbsizes_data[i];
  }
  std::vector<int64_t> nbmaps_shape({sum_nbmaps,2});
  Nbmaps = context->Output(3, nbmaps_shape);
  int* nbmaps_data = Nbmaps->template MutableData<int32_t>();
  count = 0;
  for (size_t i = 0; i < kernel_volume; i++){
    for(size_t j = 0; j < num_output_coords; j++){
      if(results[i * num_output_coords + j] != 0) {
        nbmaps_data[ count * 2 ] = results[i * num_output_coords + j] - 1;
        nbmaps_data[ count * 2 + 1] = j;
        count ++;
      }
    }
  }
  map_cycles += read_cycles() - map_start;
  parse_nbmaps_cycles += read_cycles() - parse_nbmaps_start;
  // std::cout << "debug-zxr: end buildkmap" << std::endl;
  return Status::OK();
}


template <typename T>
Status SpConv3d<T>::ConvolutionForward(const Tensor* InputFeats, Tensor* &OutputFeats, const Tensor* Weight, 
                                        const Tensor* Nbmaps, const Tensor* Nbsizes) const {

  unsigned long long conv_start = read_cycles();

  const float* input_feats_data = InputFeats->template Data<float>();
  float* output_feats_data = OutputFeats->template MutableData<float>();
  const float* weight_data = Weight->template Data<float>();
  const int* nbmaps_data = Nbmaps->template Data<int32_t>();
  const int* nbsizes_data = Nbsizes->template Data<int32_t>();
  size_t kernel_volume = conv_attrs_.kernel_shape_[0] * conv_attrs_.kernel_shape_[1] * conv_attrs_.kernel_shape_[2];

  // std::cout << "debug-zxr: start convolution_forward_cpu" << std::endl;
  convolution_forward_cpu(input_feats_data, output_feats_data, weight_data, nbmaps_data, nbsizes_data, conv_attrs_.transposed == 1, 
                          static_cast<const int>(Weight->Shape()[1]), static_cast<const int>(Weight->Shape()[2]), 
                          static_cast<const int>(InputFeats->Shape()[0]), static_cast<const int>(OutputFeats->Shape()[0]), static_cast<const int>(kernel_volume), 
                          static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode());     
  conv_cycles += read_cycles() - conv_start;
  // std::cout << "debug-zxr: end convolution_forward_cpu" << std::endl;
  return Status::OK();
}

template <typename T>
Status SpConv3d<T>::PropagateTensorDataFromInputToOutput(const Tensor* X, Tensor* Y) const {
    
  ORT_ENFORCE(X != nullptr);
  const TensorShape& shape = X->Shape();
  auto X_type = X->DataType();

  const void* source = X->DataRaw(X_type);
  void* target = Y->MutableDataRaw(X_type);
  //If source and target pointers are not equal, we need to copy the data.
  if (target != source) {
    if (!X->IsDataTypeString()) {
      memcpy(target, source, shape.Size() * X_type->Size());
    } else {
      // handle std::string
      const auto* src = X->template Data<std::string>();
      auto* dst = Y->template MutableData<std::string>();
      std::copy(src, src + shape.Size(), dst);
    }
  }
  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime

// #endif
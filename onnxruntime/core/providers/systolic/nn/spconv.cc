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
#include "core/mlas/lib/systolic/systolic_sparse.cpp" 
#include <algorithm>


#ifdef SYSTOLIC_FP32

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_KERNEL_EX(
    SpConv3d,
    kOnnxDomain,
    14,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SpConv3d<float>);

template <typename T>
Status SpConv3d<T>::Compute(OpKernelContext* context) const {
  const Tensor* InputCoords = context->Input<Tensor>(0);
  const Tensor* InputFeats = context->Input<Tensor>(1);
  const Tensor* InputStrides = context->Input<Tensor>(2);
  const Tensor* Weight = context->Input<Tensor>(3);
  const Tensor* Bias = context->Input<Tensor>(4);             // optional. nullptr if not provided
  const Tensor* OutputCoords_i = context->Input<Tensor>(5);   // optional. nullptr if not provided
  const Tensor* Nbmaps_i = context->Input<Tensor>(6);         // optional. nullptr if not provided
  const Tensor* Nbsizes_i = context->Input<Tensor>(7);        // optional. nullptr if not provided
  const Tensor* SizesIO_i = context->Input<Tensor>(8);        // optional. nullptr if not provided
    // OutputCoords_i, Nbmaps_i, Nbsizes_i, SizesIO_i should be provided togather, or none of them is provided

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
  Tensor* OutputStrides= context->Output(2, InputStrides->Shape());
  Tensor* Nbmaps_o;
  Tensor* Nbsizes_o;
  std::vector<int64_t> sizes_io_shape({1,2});
  Tensor* SizesIO_o = context->Output(5, sizes_io_shape); 

  // Tensor* OutputCoords = context->Output(0, InputCoords->Shape());
  // Tensor* OutputFeats = context->Output(1, InputFeats->Shape());
  // Tensor* OutputStrides = context->Output(2, InputStrides->Shape());
  // std::vector<int64_t> test_shape({20, 2});
  // std::vector<int64_t> test_shape1({1,2});
  // Tensor* Nbmaps_o = context->Output(3, test_shape);
  // Tensor* Nbsizes_o = context->Output(4, test_shape);
  // Tensor* SizesIO_o = context->Output(5, test_shape1); 
  const int* input_coords_data = InputCoords->template Data<int32_t>();
  const float* input_feats_data = InputFeats->template Data<float>();
  const int* input_strides_data = InputStrides->template Data<int32_t>();
  const float* weight_data = Weight->template Data<float>();

// ----------------------------------------------------------------
  // std::cout << "print input data" << std::endl;
  // std::cout << "input_coords_data:" << std::endl;
  // for (size_t i = 0; i < InputCoords->Shape()[0]; i++){
  //   for (size_t j = 0; j < InputCoords->Shape()[1]; j++){
  //     std::cout << input_coords_data[i * InputCoords->Shape()[1] + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << "input_feats_data:" << std::endl;
  // for (size_t i = 0; i < InputFeats->Shape()[0]; i++){
  //   for (size_t j = 0; j < InputFeats->Shape()[1]; j++){
  //     std::cout << input_feats_data[i * InputFeats->Shape()[1] + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << "input_strides_data:" << std::endl;
  // for (size_t i = 0; i < InputStrides->Shape()[0]; i++){
  //   std::cout << input_strides_data[i] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "weight_data:" << std::endl;
  // for (size_t i = 0; i < Weight->Shape()[0]; i++){
  //   for (size_t j = 0; j < Weight->Shape()[1]; j++){
  //     for (size_t k = 0; k < Weight->Shape()[2]; k++){
  //       std::cout << weight_data[ (i * Weight->Shape()[0] + j) * Weight->Shape()[1] + k] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  //   std::cout << std::endl;
  // }
//----------------------------------------------------------------

  if ( !transposed ){
    ORT_RETURN_IF_ERROR(BuildKmap(context, InputCoords, InputStrides, OutputCoords, Nbmaps_o, Nbsizes_o));
    int32_t* sizes_io_data = SizesIO_o -> MutableData<int32_t>();
    sizes_io_data[0] = InputCoords->Shape()[0];
    sizes_io_data[1] = OutputCoords->Shape()[0];
    std::vector<int64_t> output_feats_shape({OutputCoords->Shape()[0], InputFeats->Shape()[1]});
    OutputFeats = context->Output(1, output_feats_shape);
    ORT_RETURN_IF_ERROR(ConvolutionForward(InputFeats, OutputFeats, Weight, Nbmaps_o, Nbsizes_o));
    int* output_strides_data = OutputStrides-> template MutableData<int32_t>();
    for (size_t i = 0; i < 3; i++){
      output_strides_data[i] = input_strides_data[i] * strides[i];
    }

  }else {
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(OutputCoords_i, OutputCoords));
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(Nbmaps_i, Nbmaps_o));
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(Nbsizes_i, Nbsizes_o));
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(SizesIO_i, SizesIO_o));
    ORT_RETURN_IF_ERROR(ConvolutionForward(InputFeats, OutputFeats, Weight, Nbmaps_o, Nbsizes_o));
    int32_t* output_strides_data = OutputStrides -> template MutableData<int32_t>();
    for (size_t i = 0; i < 3; i++){
      output_strides_data[i] = input_strides_data[i] / strides[i];
    }
    std::cout << "transposed case not yet implemented" << std::endl;
  }
  return Status::OK();
}


template <typename T>
Status SpConv3d<T>::BuildKmap(OpKernelContext * context, const Tensor* InputCoords, const Tensor* InputStrides, 
                              Tensor* OutputCoords, Tensor* Nbmaps, Tensor* Nbsizes) const {
  std::vector<int64_t> kernel_shape = conv_attrs_.kernel_shape_;
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }
  // build kernel offsets
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
  std::vector<int32_t> offsets(kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * 3);
  for (size_t i = 0; i < kernel_shape[0]; i++) {
    for (size_t j = 0; j < kernel_shape[1]; j++) {
      for (size_t k = 0; k < kernel_shape[2]; k++) {
        offsets[( i * kernel_shape[1] * kernel_shape[2] + j * kernel_shape[2] + k ) * 3] = x_dim_offsets[i] ;
        offsets[( i * kernel_shape[1] * kernel_shape[2] + j * kernel_shape[2] + k ) * 3 + 1] = y_dim_offsets[j];
        offsets[( i * kernel_shape[1] * kernel_shape[2] + j * kernel_shape[2] + k ) * 3 + 2] = z_dim_offsets[k];
      }
    }
  }

  const std::vector<int32_t> input_coords_tensor(InputCoords-> template Data<int32_t>()); // ?
  std::vector<int64_t> references = hash_cpu(input_coords_tensor);

  if ( kernel_shape[0] > 1 || kernel_shape[1] > 1 || kernel_shape[2] > 1) {
    //downsampling
    std::vector<int32_t> sample_stride(3); 
    for (size_t i = 0; i < 3; i++) {
      sample_stride[i] = strides[i] * input_strides_data[i];
    }
    const TensorShape& input_coords_shape = InputCoords -> Shape();
    std::vector<std::vector<int32_t>> output_coords_vector_uncoalesced();
    output_coords_vector_uncoalesced.resize(input_coords_shape[0], std::vector<int32_t>(4));
    for (size_t i = 0; i < input_coords_shape[0]; i++) {
      output_coords_vector_uncoalesced[ i * 4 ] = input_coords_data[ i * 4 ] / sample_stride[0] * sample_stride[0];
      output_coords_vector_uncoalesced[ i * 4 + 1 ] = input_coords_data[ i * 4 + 1 ] / sample_stride[1] * sample_stride[1];
      output_coords_vector_uncoalesced[ i * 4 + 2 ] = input_coords_data[ i * 4 + 2 ] / sample_stride[2] * sample_stride[2];
      output_coords_vector_uncoalesced[ i * 4 + 3 ] = input_coords_data[ i * 4 + 3 ];
    }
    bool cmp(std::vector<int32_t> &a, std::vector<int32_t> &b){
      if(a[3] != b[3]){
        return a[3] < b[3];
      } else if(a[0] != b[0]){
        return a[0] < b[0];
      } else if(a[1] != b[1]){
        return a[1] < b[1];
      } else if(a[2] != b[2]){
        return a[2] < b[2];
      }
    }
    std::sort(output_coords_vector_uncoalesced.begin(), output_coords_vector_uncoalesced.end(), cmp);
    auto pos = std::unique(output_coords_vector_uncoalesced.begin(), output_coords_vector_uncoalesced.end());
    output_coords_vector_uncoalesced.erase(pos, output_coords_vector_uncoalesced.end());

    size_t num_output_coords = output_coords_vector_uncoalesced.size();
    std::vector<int32_t> output_coords_shape({num_output_coords, 4});
    OutputCoords = context->Output(0, output_coords_shape);
    int32_t* output_coords_data = OutputCoords -> template MutableData<int32_t>();
    for (size_t i = 0; i < num_output_coords; ++i){
      output_coords_data[i * 4] = output_coords_vector_uncoalesced[i][0];
      output_coords_data[i * 4 + 1] = output_coords_vector_uncoalesced[i][1];
      output_coords_data[i * 4 + 2] = output_coords_vector_uncoalesced[i][2];
      output_coords_data[i * 4 + 3] = output_coords_vector_uncoalesced[i][3];
    }
    const std::vector<int32_t> output_coords_vector(output_coords_data, num_output_coords * 4);
  } else {
    // use input coordinates as output coordinates
    OutputCoords = context->Output(0, InputCoords->Shape());
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(InputCoords, OutputCoords));
    const int32_t* output_coords_data = OutputCoords -> template Data<int32_t>();
    const std::vector<int32_t> output_coords_vector(output_coords_data, num_output_coords * 4);
  }
  std::vector<int64_t> queries = kernel_hash_cpu(output_coords_vector, offsets);
  std::vector<int64_t> results =  hash_query_cpu(queries, references);
  //parse nbsizes
  size_t kernel_volume = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
  std::vector<int64_t> nbsizes_shape({kernel_volume});
  Nbsizes = context->Output(4, nbsizes_shape);
  int32_t* nbsizes_data = Nbsizes->template MutableData<int32_t>();
  for (size_t i = 0; i < kernel_volume; i++){
    size_t sum = 0;
    for(size_t j = 0; j < num_output_coords; j++){
      if(results[i * num_output_coords + j] != 0) {
        sum++;
      }
    }
    nbsizes_data[i] = sum;
  }
  //parse nbmaps
  size_t sum_nbmaps = 0;
  for(size_t i = 0; i < kernel_volume; i++){
    sum_nbmaps += nbsizes_data[i];
  }
  std::vector<int64_t> nbmaps_shape({sum_nbmaps,2});
  Nbmaps = context->Output(3, nbsizes_shape);
  int32_t* nbmaps_data = Nbmaps->template MutableData<int32_t>();
  size_t count = 0;
  for (size_t i = 0; i < kernel_volume; i++){
    for(size_t j = 0; j < num_output_coords; j++){
      if(results[i * num_output_coords + j] != 0) {
        nbmaps_data[ i * 2 ] = results[i * num_output_coords + j] - 1;
        nbmaps_data[ i * 2 ] = j;
      }
    }
  }
  return Status::OK();
}


template <typename T>
Status SpConv3d<T>::ConvolutionForward(const Tensor* InputFeats, Tensor* OutputFeats, const Tensor* Weight, 
                                        const Tensor* Nbmaps, const Tensor* Nbsizes) const {

  const float* input_feats_data = InputFeats->template Data<float>();
  float* output_feats_data = OutputFeats->template MutableData<float>();
  const float* weight_data = Weight->template Data<float>();
  const int32_t* nbmaps_data = Nbmaps->template Data<int32_t>();
  const int32_t* nbsizes_data = Nbsizes->template Data<int32_t>();
  size_t kernel_volume = conv_attrs_.kernel_shape_[0] * conv_attrs_.kernel_shape_[1] * conv_attrs_.kernel_shape_[2];
  convolution_forward_cpu(input_feats_data, output_feats_data, weight_data, nbmaps_data, nbsizes_data, 
                          const(conv_attrs_.transposed == 1), static_cast<int>(InputFeats->Shape()[0]), 
                          static_cast<int>(OutputFeats->Shape()[0]), , WS);                           
  return Status::OK();
}

template <typename T>
Status SpConv3d<T>::PropagateTensorDataFromInputToOutput(const Tensor* X, Tensor* Y) const {
  // auto Input_ml_type = context->InputType(0);
  // if (Input_ml_type != DataTypeImpl::GetType<Tensor>()) {
  //   return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "propagateTensorDataFromInputToOutput expects a Tensor, got other type.");
  // } 
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

#endif
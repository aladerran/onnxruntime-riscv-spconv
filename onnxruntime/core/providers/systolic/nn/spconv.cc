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
  const Tensor* Nbmaps_i = context->Input<Tensor>(4);         // optional. nullptr if not provided
  const Tensor* Nbsizes_i = context->Input<Tensor>(5);        // optional. nullptr if not provided
  const Tensor* SizesIO_i = context->Input<Tensor>(6);        // optional. nullptr if not provided
  const Tensor* OutputCoords_i = context->Input<Tensor>(7);   // optional. nullptr if not provided
  const Tensor* Bias = context->Input<Tensor>(8);             // optional. nullptr if not provided

    // OutputCoords_i, Nbmaps_i, Nbsizes_i, SizesIO_i should be provided togather, or none of them is provided

  std::cout << "debug-zxr: start spconv3d computing" << std::endl;
  std::cout << "debug-zxr:domain:" << context->GetOpDomain() << " /type:" << context->GetOpType() << " /name:" << context->GetNodeName() << std::endl;
  std::string node_name = context->GetNodeName();
  bool printflag = node_name.find("/upsample/conv3d/SpConv3d")!=std::string::npos;


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
  if(printflag){
    std::cout << "print input data" << std::endl;
    std::cout << "input_coords_data:" << std::endl;
    for (size_t i = 0; i < InputCoords->Shape()[0]; i++){
      for (size_t j = 0; j < InputCoords->Shape()[1]; j++){
        std::cout << input_coords_data[i * InputCoords->Shape()[1] + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "input_feats_data:" << std::endl;
    for (size_t i = 0; i < InputFeats->Shape()[0] ; i++){
      for (size_t j = 0; j < InputFeats->Shape()[1]; j++){
        std::cout << input_feats_data[i * InputFeats->Shape()[1] + j] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "input_strides_data:" << std::endl;
    for (size_t i = 0; i < InputStrides->Shape()[0]; i++){
      std::cout << input_strides_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "weight_data:" << std::endl;
    for (size_t i = 0; i < Weight->Shape()[0] && i < 27; i++){
      for (size_t j = 0; j < Weight->Shape()[1] && j < 10; j++){
        for (size_t k = 0; k < Weight->Shape()[2] && k < 10; k++){
          std::cout << weight_data[ (i * Weight->Shape()[1] + j) * Weight->Shape()[2] + k] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
//----------------------------------------------------------------

  if ( !transposed ){
    std::cout << "debug-zxr: start buildkmap" << std::endl;
    ORT_RETURN_IF_ERROR(BuildKmap(context, InputCoords, InputStrides, OutputCoords, Nbmaps_o, Nbsizes_o));
    std::cout << "OutputCoords:" << OutputCoords << "; &OutputCoords:" << &OutputCoords << std::endl;
    std::cout << "Nbsizes:" << Nbsizes_o << "; &Nbsizes:" << &Nbsizes_o << std::endl;
    std::cout << "Nbmaps:" << Nbmaps_o << "; &Nbmaps:" << &Nbmaps_o << std::endl;
    int32_t* sizes_io_data = SizesIO_o -> MutableData<int32_t>();
    std::cout << "debug-zxr: @2" << std::endl;
    sizes_io_data[0] = static_cast<int32_t>(InputCoords->Shape()[0]);
    std::cout << "debug-zxr: @3" << std::endl;
    sizes_io_data[1] = static_cast<int32_t>(OutputCoords->Shape()[0]);
    std::cout << "debug-zxr: @4" << std::endl;
    std::vector<int64_t> output_feats_shape({OutputCoords->Shape()[0], Weight->Shape()[2]});
    std::cout << "debug-zxr: @5" << std::endl;
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

    if(printflag) {
      std::cout << " output coordinates: " << std::endl;
      const int* output_coords_data = OutputCoords_i->template Data<int32_t>();
      for (size_t i = 0; i < OutputCoords_i->Shape().Size(); i+=4){
        std::cout << output_coords_data[i] << " " << output_coords_data[i+1] << " " << output_coords_data[i+2] << " " << output_coords_data[i+3] << std::endl;
      }
      std::cout << " nbmaps: " << std::endl;
      const int* nbmaps_data = Nbmaps_i->template Data<int32_t>();
      for (size_t i = 0; i < Nbmaps_i->Shape()[0]; i++){
        std::cout << "[" << nbmaps_data[i * 2] << "," << nbmaps_data[i * 2 + 1] << "] ";
      }
      std::cout << std::endl;
      std::cout << " nbsizes: " << std::endl;
      const int* nbsizes_data = Nbsizes_i->template Data<int32_t>();
      for (size_t i = 0; i < Nbsizes_i->Shape()[0]; i++){
        std::cout << nbsizes_data[i] << " ";
      }
      std::cout << std::endl;
    }
    


    std::vector<int64_t> output_feats_shape({OutputCoords->Shape()[0], Weight->Shape()[2]});
    OutputFeats = context->Output(1, output_feats_shape);
    ORT_RETURN_IF_ERROR(ConvolutionForward(InputFeats, OutputFeats, Weight, Nbmaps_o, Nbsizes_o));
    int32_t* output_strides_data = OutputStrides -> template MutableData<int32_t>();
    for (size_t i = 0; i < 3; i++){
      output_strides_data[i] = input_strides_data[i] / strides[i];
    }
  }

  // int* output_coords_data = OutputCoords -> template MutableData<int32_t>();
  // const float* output_feats_data = OutputFeats->template Data<float>();
  // const int* output_strides_data = OutputStrides->template Data<int32_t>();
  // std::cout << "print outputs:" << std::endl;
  // std::cout << "output_coords_data:" << std::endl;
  // for (size_t i = 0; i < OutputCoords->Shape()[0]; i++){
  //   for (size_t j = 0; j < OutputCoords->Shape()[1]; j++){
  //     std::cout << output_coords_data[i * OutputCoords->Shape()[1] + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << "output_feats_data:" << std::endl;
  // for (size_t i = 0; i < OutputFeats->Shape()[0]; i++){
  //   for (size_t j = 0; j < OutputFeats->Shape()[1]; j++){
  //     std::cout << output_feats_data[i * OutputFeats->Shape()[1] + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << "output_strides_data:" << std::endl;
  // for (size_t i = 0; i < OutputStrides->Shape()[0]; i++){
  //   std::cout << output_strides_data[i] << " ";
  // }
  // const int* nbmaps_data = Nbmaps_o -> template Data<int32_t>();
  // const int* nbsizes_data = Nbsizes_o -> template Data<int32_t>();
  // std::cout << "nbmaps_data:" << std::endl;
  // for (size_t i = 0; i < Nbmaps_o -> Shape()[0]; i++){
  //   std::cout << nbmaps_data[i * 2] << " " << nbmaps_data[i * 2 + 1] << std::endl;
  // }
  // std::cout << "nbsizes_data:" << std::endl;
  // for(size_t i = 0; i < Nbsizes_o->Shape()[0]; i++){
  //   std::cout << nbsizes_data[i] << " ";
  // }
  // std::cout << std::endl;


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
  int64_t kernel_volume = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
  std::vector<int32_t> offsets(kernel_volume * 3);
  for (size_t i = 0; i < kernel_shape[0]; i++) {
    for (size_t j = 0; j < kernel_shape[1]; j++) {
      for (size_t k = 0; k < kernel_shape[2]; k++) {
        offsets[( i * kernel_shape[1] * kernel_shape[2] + j * kernel_shape[2] + k ) * 3] = x_dim_offsets[i] ;
        offsets[( i * kernel_shape[1] * kernel_shape[2] + j * kernel_shape[2] + k ) * 3 + 1] = y_dim_offsets[j];
        offsets[( i * kernel_shape[1] * kernel_shape[2] + j * kernel_shape[2] + k ) * 3 + 2] = z_dim_offsets[k];
      }
    }
  }

  const TensorShape& input_coords_shape = InputCoords -> Shape();
  const int* input_coords_data = InputCoords-> template Data<int32_t>();
  std::vector<int64_t> references(input_coords_shape[0]);
  hash_cpu(input_coords_data, references.data(), input_coords_shape[0]);
  //debug-zxr
  // if(conv_attrs_.printflag){
  //   std::cout << "References:" << std::endl;
  //   for (size_t i = 0; i < references.size(); i++){
  //     std::cout << references[i] << " ";
  //   }
  // }
  

  int* output_coords_data;
  int64_t num_output_coords;
  if ( conv_attrs_.strides[0] > 1 || conv_attrs_.strides[1] > 1 || conv_attrs_.strides[2] > 1) {
    //downsampling
    const int* input_strides_data = InputStrides->template Data<int32_t>();
    std::vector<int32_t> sample_stride(3); 
    for (size_t i = 0; i < 3; i++) {
      sample_stride[i] = strides[i] * input_strides_data[i];
    }

    std::vector<std::vector<int32_t>> output_coords_vector_uncoalesced(input_coords_shape[0], std::vector<int32_t>(4));
    for (size_t i = 0; i < input_coords_shape[0]; i++) {
      output_coords_vector_uncoalesced[i][0] = input_coords_data[ i * 4 ] / sample_stride[0] * sample_stride[0];
      output_coords_vector_uncoalesced[i][1] = input_coords_data[ i * 4 + 1 ] / sample_stride[1] * sample_stride[1];
      output_coords_vector_uncoalesced[i][2] = input_coords_data[ i * 4 + 2 ] / sample_stride[2] * sample_stride[2];
      output_coords_vector_uncoalesced[i][3] = input_coords_data[ i * 4 + 3 ];
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
  //debug-zxr
  // std::cout << " output coordinates vector: " << std::endl;
  // for (size_t i = 0; i < output_coords_vector.size(); i+=4){
  //   std::cout << output_coords_vector[i] << " " << output_coords_vector[i+1] << " " << output_coords_vector[i+2] << " " << output_coords_vector[i+3] << std::endl;
  // }
  std::vector<int64_t> queries(num_output_coords * kernel_volume);
  kernel_hash_cpu(output_coords_data, offsets.data(), queries.data(), num_output_coords, kernel_volume);
  // std::cout << "queries:" << std::endl;
  // for (size_t i = 0; i < queries.size(); i++){
  //   std::cout << queries[i] << " ";
  // }

  std::vector<int64_t> indices(num_output_coords);
  std::iota(indices.begin(), indices.end(), 0);
  std::vector<int64_t> results(queries.size());
  hash_query_cpu(queries.data(), references.data(), indices.data(), results.data(), references.size(), queries.size());

  // std::cout << "results:" << std::endl;
  // for (size_t i = 0; i < results.size(); i++){
  //   std::cout << results[i] << " ";
  // }

  //parse nbsizes
  //int64_t kernel_volume = kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
  std::vector<int64_t> nbsizes_shape({kernel_volume});
  Nbsizes = context->Output(4, nbsizes_shape);
  std::cout << "debug-zxr: parse nbsizes" << std::endl;
  int* nbsizes_data = Nbsizes->template MutableData<int32_t>();
  for (size_t i = 0; i < kernel_volume; i++){
    size_t sum = 0;
    for(size_t j = 0; j < num_output_coords; j++){
      if(results[i * num_output_coords + j] != 0) {
        sum++;
      }
    }
    nbsizes_data[i] = sum;
  }

  std::cout << "nbsizes: " << std::endl;
  for (size_t i = 0; i < kernel_volume; i++){
    std::cout << nbsizes_data[i] << " ";
  }


  //parse nbmaps
  std::cout << "debug-zxr: parse nbmaps" << std::endl;
  int64_t sum_nbmaps = 0;
  for(size_t i = 0; i < kernel_volume; i++){
    sum_nbmaps += nbsizes_data[i];
  }
  std::vector<int64_t> nbmaps_shape({sum_nbmaps,2});
  std::cout << "Nbmaps:" << Nbmaps << "; &Nbmaps:" << &Nbmaps << std::endl;
  Nbmaps = context->Output(3, nbmaps_shape);
  std::cout << "Nbmaps:" << Nbmaps << "; &Nbmaps:" << &Nbmaps << std::endl;
  int* nbmaps_data = Nbmaps->template MutableData<int32_t>();
  size_t count = 0;
  for (size_t i = 0; i < kernel_volume; i++){
    for(size_t j = 0; j < num_output_coords; j++){
      if(results[i * num_output_coords + j] != 0) {
        nbmaps_data[ count * 2 ] = results[i * num_output_coords + j] - 1;
        nbmaps_data[ count * 2 + 1] = j;
        count ++;
      }
    }
  }
  // std::cout << "nbmaps: " << std::endl;
  // for (size_t i = 0; i < sum_nbmaps; i++){
  //   std::cout << nbmaps_data[i * 2] << " " << nbmaps_data[i * 2 + 1] << std::endl;
  // }
  std::cout << "debug-zxr: end buildkmap" << std::endl;
  return Status::OK();
}


template <typename T>
Status SpConv3d<T>::ConvolutionForward(const Tensor* InputFeats, Tensor* &OutputFeats, const Tensor* Weight, 
                                        const Tensor* Nbmaps, const Tensor* Nbsizes) const {

  const float* input_feats_data = InputFeats->template Data<float>();
  float* output_feats_data = OutputFeats->template MutableData<float>();
  const float* weight_data = Weight->template Data<float>();
  const int* nbmaps_data = Nbmaps->template Data<int32_t>();
  const int* nbsizes_data = Nbsizes->template Data<int32_t>();
  size_t kernel_volume = conv_attrs_.kernel_shape_[0] * conv_attrs_.kernel_shape_[1] * conv_attrs_.kernel_shape_[2];

  // std::vector<float> input_feats_vector(input_feats_data, input_feats_data + InputFeats->Shape().Size());
  // std::vector<float> output_feats_vector(output_feats_data, output_feats_data + OutputFeats->Shape().Size());
  // std::vector<float> weight_vector(weight_data, weight_data + Weight->Shape().Size());
  // std::vector<int> nbmaps_vector(nbmaps_data, nbmaps_data + Nbmaps->Shape().Size());
  // std::vector<int> nbsizes_vector(nbsizes_data, nbsizes_data + Nbsizes->Shape().Size());

  std::cout << "debug-zxr: start convolution_forward_cpu" << std::endl;
  convolution_forward_cpu(input_feats_data, output_feats_data, weight_data, nbmaps_data, nbsizes_data, conv_attrs_.transposed == 1, 
                          static_cast<const int>(Weight->Shape()[1]), static_cast<const int>(Weight->Shape()[2]), 
                          static_cast<const int>(InputFeats->Shape()[0]), static_cast<const int>(OutputFeats->Shape()[0]), static_cast<const int>(kernel_volume), 
                          static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode());                   
  std::cout << "debug-zxr: end convolution_forward_cpu" << std::endl;
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
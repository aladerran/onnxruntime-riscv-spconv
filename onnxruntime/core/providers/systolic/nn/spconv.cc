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

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateSpInputShape(InputCoords, InputFeats, InputStrides, Weight));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeSpKernelShape(Weight->Shape(), kernel_shape));

  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }
  const bool transposed = conv_attrs_.transposed == 1 ? 1 : 0 ; 


  Tensor* OutputCoords = context->Output(0, InputCoords->Shape());
  Tensor* OutputFeats = context->Output(1, InputFeats->Shape());
  Tensor* OutputStrides = context->Output(2, InputStrides->Shape());
  std::vector<int64_t> test_shape({20, 2});
  std::vector<int64_t> test_shape1({1,2});
  Tensor* Nbmaps_o = context->Output(3, test_shape);
  Tensor* Nbsizes_o = context->Output(4, test_shape);
  Tensor* SizesIO_o = context->Output(5, test_shape1); 

  

  const long* input_coords_data = InputCoords->template Data<int64_t>();
  const float* input_feats_data = InputFeats->template Data<float>();
  const long* input_strides_data = InputStrides->template Data<int64_t>();
  const float* weight_data = Weight->template Data<float>();

  

// ----------------------------------------------------------------

  std::cout << "print input data" << std::endl;
  std::cout << "input_coords_data:" << std::endl;
  for (size_t i = 0; i < InputCoords->Shape()[0]; i++){
    for (size_t j = 0; j < InputCoords->Shape()[1]; j++){
      std::cout << input_coords_data[i * InputCoords->Shape()[1] + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "input_feats_data:" << std::endl;
  for (size_t i = 0; i < InputFeats->Shape()[0]; i++){
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
  for (size_t i = 0; i < Weight->Shape()[0]; i++){
    for (size_t j = 0; j < Weight->Shape()[1]; j++){
      for (size_t k = 0; k < Weight->Shape()[2]; k++){
        std::cout << weight_data[ (i * Weight->Shape()[0] + j) * Weight->Shape()[1] + k] << " " << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << conv_attrs_.

//----------------------------------------------------------------

  if ( !transposed ){
    ORT_RETURN_IF_ERROR(BuildKmap(InputCoords, InputStrides, Nbmaps_o, Nbsizes_o, OutputCoords));
    int64_t size_i = InputCoords->Shape()[0];
    int64_t size_o = OutputCoords->Shape()[0];
    const long* sizes_io_data = SizesIO_o -> MutableData<int64_t>();
    size_io_data[0] = size_i;
    size_io_data[1] = size_o;
    ORT_RETURN_IF_ERROR(ConvolutionForward(OutputFeats, InputFeats, Weight, Nbmaps_o, Nbsizes_o, SizesIO_o));



  }else {
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(OutputCoords_i, OutputCoords));
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(Nbmaps_i, Nbmaps_o));
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(Nbsizes_i, Nbsizes_o));
    ORT_RETURN_IF_ERROR(PropagateTensorDataFromInputToOutput(SizesIO_i, SizesIO_o));
    ORT_RETURN_IF_ERROR(ConvolutionForward(OutputFeats, InputFeats, Weight, Nbmaps_o, Nbsizes_o, SizesIO_o));



    std::cout << "transposed case not yet implemented" << std::endl;
  }





  return Status::OK();
}



template <typename T>
Status SpConv3d<T>::BuildKmap(const Tensor* InputCoords, const Tensor* InputStrides, 
                              const Tensor* Nbmaps, const Tensor* Nbsizes, 
                              const Tensor* OutputCoords) const {
  //TODO
  
  return Status::OK();
}

template <typename T>
Status SpConv3d<T>::ConvolutionForward(const Tensor* OutputFeats, const Tensor* InputFeats, const Tensor* Weight, 
                                        const Tensor* Nbmaps, const Tensor* Nbsizes, const Tensor* Sizes_IO) const {
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
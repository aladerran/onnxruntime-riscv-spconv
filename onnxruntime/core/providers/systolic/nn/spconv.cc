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
  const auto* InputCoords = context->Input<Tensor>(0);
  const auto* InputFeats = context->Input<Tensor>(1);
  const auto* InputStrides = context->Input<Tensor>(2);
  const auto* Weight = context->Input<Tensor>(3);
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
  const bool transposed = conv_attrs_.transposed == 1 ? 1 : 0 ; 


  Tensor* OutputCoords = context->Output(0, InputCoords->Shape());
  Tensor* OutputFeasts = context->Output(1, InputFeats->Shape());
  Tensor* OutputStrides = context->Output(2, InputStrides->Shape());
  std::vector<int64_t> test_shape = {20, 2};
  std::vector<int64_t> test_shape1 = {1,2};
  Tensor* Nbmaps_o = context->Output(3, test_shape);
  Tensor* Nbsizes_o = context->Output(4, test_shape);
  Tensor* SizesIO_o = context->Output(3, test_shape1); 


  return Status::OK();
}

}  // namespace systolic
}  // namespace onnxruntime

#endif
#include "bn_relu.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/common/safeint.h"
#include "conv_pool_helper.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/nn/batch_norm_helper.h"
#include <safeint/SafeInt.hpp>


using onnxruntime::concurrency::ThreadPool;

unsigned long long systolic_norm_relu_cycles;

namespace onnxruntime {
namespace systolic {



ONNX_OPERATOR_TYPED_KERNEL_EX(
    SystolicBatchNormRelu,
    kOnnxDomain,
    14,
    float,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SystolicBatchNormRelu<float>);

template <typename T>
Status SystolicBatchNormRelu<T>::Compute(OpKernelContext* p_op_kernel_context) const {

  std::cout << "debug-lsr:domain:" << p_op_kernel_context->GetOpDomain() << " /type:" << p_op_kernel_context->GetOpType() << " /name:" << p_op_kernel_context->GetNodeName() << std::endl;

  unsigned long long norm_start = read_cycles();

  const auto* X = p_op_kernel_context->Input<Tensor>(0);
  const auto* scale = p_op_kernel_context->Input<Tensor>(1);
  const auto* B = p_op_kernel_context->Input<Tensor>(2);
  const auto* mean = p_op_kernel_context->Input<Tensor>(3);
  const auto* var = p_op_kernel_context->Input<Tensor>(4);

  ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, is_spatial_));

  const TensorShape& x_shape = X->Shape();
  Tensor* Y = p_op_kernel_context->Output(0, x_shape);

  const auto& dims_vec = x_shape.GetDims();
  const size_t N = dims_vec[0];
  const size_t C = dims_vec[1];  // assume NC as per the spec

  if(!is_train_){

    const T* X_pointer = X->template Data<T>();
    T* Y_pointer = Y->template MutableData<T>();
    const T* B_pointer = B->template Data<T>();

    // we assuem var !=0 at inference session so that we can offload it to Systolic Array
    // create an adjusted_scale in dim [C, C] from scale by copying scale into a 2-D vector
    std::vector<float> scale_vector(C * C);
    for (size_t i = 0; i < C; ++i) {
      scale_vector[i * C + i] = scale->template Data<T>()[i];
    }

    // call SystolicMutilply
    SystolicMultiply(
        /* accelerator_mode */ static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode(), 
        /* relu */ true, 
        /* dimI */ N,
        /* dimJ */ C,
        /* dimK */ C,
        /* in1 */ X_pointer,
        /* strideIn1 */ C,
        /* in2 */ scale_vector.data(), 
        /* strideIn2 */ C,
        /* out */ Y_pointer,
        /* strideOut */ C,
        /* real_multiplier */ 1.0f,
        /* bias */ B_pointer,
        /* strideBias */ C,
        /* repeating_bias */ true);

  }

  systolic_norm_relu_cycles += read_cycles() - norm_start;
  if(p_op_kernel_context->GetNodeName() == "/fuse/resblock2/main/batchnorm1/norm_3/BatchNormalization_SystolicBatchNormRelu"
   || p_op_kernel_context->GetNodeName() == "/4/4.0/batchnorm/norm/BatchNormalization_SystolicBatchNormRelu"){
    std::cout << "Systolic BatchNorm + Relu cycles: " << systolic_norm_relu_cycles << std::endl;
  }
  

  return Status::OK();
}


} // namespace systolic
}  // namespace onnxruntime
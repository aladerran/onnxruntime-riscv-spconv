#include "batch_norm.h"
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



namespace onnxruntime {
namespace systolic {


#ifdef SYSTOLIC_FP32

ONNX_OPERATOR_TYPED_KERNEL_EX(
    BatchNormalization,
    kOnnxDomain,
    14,
    float,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
BatchNorm<float>);


#endif

template <typename T>
Status BatchNorm<T>::Compute(OpKernelContext* p_op_kernel_context) const {
  //debug-zxr
  std::cout << " in BatchNorm" << std::endl;
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
  const size_t C = dims_vec[1];  // assume NCHW as per the spec

  // calculate sample_size (per individual channel)
  size_t sample_size = 1;
  for (size_t i = 2; i < dims_vec.size(); ++i) {
    sample_size *= gsl::narrow<size_t>(dims_vec[i]);
  }

  // calculate sample_size (including all channels)
  size_t sample_size_incl_all_channels = sample_size * C;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(p_op_kernel_context->GetTempSpaceAllocator(&alloc));
  
  // Saved mean corresponds to the mean from this batch
  // If these optional outputs are present (opset <= 9 or internal BN op) we re-use the space for calculations
  // Note that with opset <= 9 we will be outputting saved_inv_std_dev instead of saved_var
  Tensor *saved_mean = is_train_ ? p_op_kernel_context->Output(3, mean->Shape()) : nullptr;
  Tensor *saved_inv_std = is_train_ ? p_op_kernel_context->Output(4, var->Shape()) : nullptr;
  // With opset <= 9, both must be defined in training. If opset >= 14, neither should be defined in training
  ORT_ENFORCE(!is_train_ || ((!saved_mean && !saved_inv_std) || (saved_mean && saved_inv_std)), "Invalid number of outputs for BN training");
  Tensor saved_mean_allocated, saved_inv_std_allocated;
  if (is_train_ && !saved_mean) {
    saved_mean_allocated = Tensor(DataTypeImpl::GetType<T>(), mean->Shape(), alloc);
    saved_inv_std_allocated = Tensor(DataTypeImpl::GetType<T>(), var->Shape(), alloc);
    saved_mean = &saved_mean_allocated;
    saved_inv_std = &saved_inv_std_allocated;
  }
  ConstEigenArrayMap<T> X_arr(X->template Data<T>(),
                              is_spatial_ ? sample_size : sample_size_incl_all_channels,
                              is_spatial_ ? N * C : N);

  

  //debug-zxr
  // std::vector<float> x_vector(X->template Data<T>(), X->template Data<T>() + x_shape.Size());
  // for(int i = 0; i < x_shape.Size(); i++){
  //   std::cout << x_vector[i] << " ";
  // }
  // std::cout << "=======================" << std::endl;
  // for (int i = 0; i < sample_size; ++i) {
  //   for (int j = 0; j < N * C ; ++j) {
  //     std::cout << X_arr(i,j) << " ";
  //   }
  //   std::cout << std::endl;
  // }


  ConstEigenVectorArrayMap<T> scale_arr(scale->template Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);
  ConstEigenVectorArrayMap<T> bias_arr(B->template Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);

  // Note that we only support spatial BN for training
  if (is_train_) {
    EigenVectorArrayMap<T> saved_mean_arr(saved_mean->template MutableData<T>(), C);
    // We first calculate saved_var then later take inverse square root to get saved_inv_std
    EigenVectorArrayMap<T> saved_var_arr(saved_inv_std->template MutableData<T>(), C);
    saved_mean_arr.setZero();
    saved_var_arr.setZero();

    for (size_t nc = 0; nc < N * C; ++nc) {
      saved_mean_arr(nc % C) += X_arr.col(nc).sum();
    }

    saved_mean_arr /= static_cast<T>(N * sample_size);
    for (size_t nc = 0; nc < N * C; ++nc) {
      saved_var_arr(nc % C) += (X_arr.col(nc) - saved_mean_arr(nc % C)).matrix().squaredNorm();
    }
    saved_var_arr /= static_cast<T>(N * sample_size);

    // The running mean corresponds to the mean from all the batches
    // During inference this running mean is used as the mean for BN
    auto* running_mean = p_op_kernel_context->Output(1, mean->Shape());
    auto* running_var = p_op_kernel_context->Output(2, var->Shape());
    const auto* input_running_mean = p_op_kernel_context->Input<Tensor>(3);
    const auto* input_running_var = p_op_kernel_context->Input<Tensor>(4);

    // Assume that running mean and variance are initialized properly in the model given to us
    // Because we alias it, we have the past history here
    EigenVectorArrayMap<T> running_mean_arr(
        running_mean->template MutableData<T>(), C);
    EigenVectorArrayMap<T> running_var_arr(
        running_var->template MutableData<T>(), C);
    ConstEigenVectorArrayMap<T> input_running_mean_arr(
        input_running_mean->template Data<T>(), C);
    ConstEigenVectorArrayMap<T> input_running_var_arr(
        input_running_var->template Data<T>(), C);
    running_mean_arr = input_running_mean_arr * momentum_ + saved_mean_arr * (1. - momentum_);
    running_var_arr = input_running_var_arr * momentum_ + saved_var_arr * (1. - momentum_);
  }

  // Regardless of training or testing, we will apply the estimated mean
  // and standard deviation to the input. For testing, they are
  // specified directly by the input, and for training, they are computed
  // by the op.
  Eigen::Array<T, Eigen::Dynamic, 1> inv_std(is_spatial_ ? C : sample_size_incl_all_channels);

  if (!is_train_) {
    ConstEigenVectorArrayMap<T> var_arr(var->template Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);
    inv_std = (var_arr + epsilon_).sqrt().inverse();
  } else {
    EigenVectorArrayMap<T> saved_inv_std_arr(saved_inv_std->template MutableData<T>(), C);
    saved_inv_std_arr = (saved_inv_std_arr + epsilon_).inverse().sqrt();
    inv_std = saved_inv_std_arr;
  }

  // If we're training, do batch normalization based on computation from this batch
  ConstEigenVectorArrayMap<T> mean_arr(
      !is_train_ ? mean->template Data<T>() : saved_mean->template Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);

  // We can fuse the output computation as follows:
  //   ((x - est_mean) * (inv_var) * scale + bias
  // to
  //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)

  Eigen::Array<T, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
  Eigen::Array<T, Eigen::Dynamic, 1> new_bias = bias_arr - mean_arr * new_scale;
  EigenArrayMap<T> Y_arr(Y->template MutableData<T>(),
                          is_spatial_ ? sample_size : sample_size_incl_all_channels,
                          is_spatial_ ? N * C : N);
  const T* X_pointer = X->template Data<T>();
  T* Y_pointer = Y->template MutableData<T>();
  T* new_scale_pointer = new_scale.data();
  T* new_bias_pointer = new_bias.data();
  if (is_spatial_) {  // spatial == 1
    // std::vector<int64_t> bias_shape({static_cast<size_t>(sample_size)});
    // Tensor bias_tensor = Tensor(DataTypeImpl::GetType<T>(), bias_shape, alloc);
    std::vector<T> bias_vec(sample_size);
    for (size_t nc = 0; nc < N * C; ++nc) {
      std::fill(bias_vec.begin(), bias_vec.end(), new_bias(nc % C));
      SystolicMultiply(static_cast<const SystolicExecutionProvider*>(
                    this->Info().GetExecutionProvider())->GetAcceleratorMode(),
                    /* relu= */ false, 1, sample_size, 1,
                    new_scale_pointer + (nc % C), 
                    X_pointer + nc * sample_size, 
                    Y_pointer + nc * sample_size,
                    /*real_multiplier=*/ 1, bias_vec.data());
    }

    // std::vector<T> bias_vec(C * sample_size);
    // std::vector<T> scale_vec(C * C, 0);
    // for (size_t i = 0; i < C; ++i) {
    //   scale_vec[i * C + i] = new_scale(i);
    // }
    // for (size_t i = 0; i < C; ++i) {
    //   std::fill(bias_vec.begin() + i * sample_size, bias_vec.begin() + i * sample_size + sample_size, new_bias(i));
    // }
    // for (size_t n = 0; n < N; ++n) {
    //   SystolicMultiply(static_cast<const SystolicExecutionProvider*>(
    //                 this->Info().GetExecutionProvider())->GetAcceleratorMode(),
    //                 /* relu= */ false, C, sample_size, C,
    //                 scale_vec.data(), 
    //                 X_pointer + n * C * sample_size, 
    //                 Y_pointer + n * C * sample_size,
    //                 /*real_multiplier=*/ 1, bias_vec.data());
    // }

    // for (size_t nc = 0; nc < N * C; ++nc) {
    //   Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
    // }
  } else {  // spatial == 0
    for (size_t n = 0; n < N; ++n) {
      Y_arr.col(n) = X_arr.col(n) * new_scale.col(0) + new_bias.col(0);
    }
  }
  return Status::OK();
}


} // namespace systolic
}  // namespace onnxruntime
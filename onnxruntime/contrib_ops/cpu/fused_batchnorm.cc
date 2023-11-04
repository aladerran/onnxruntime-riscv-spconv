#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/batch_norm.h"
#include "contrib_ops/cpu/fused_activation.h"
#include "core/mlas/inc/mlas.h"


namespace onnxruntime {
namespace contrib {

template <typename T>
class FusedBatchNorm final : public BatchNorm<T> {
 public:
  FusedBatchNorm(const OpKernelInfo& info) : BatchNorm<T>(info) {
    ORT_ENFORCE(GetFusedActivationAttr(info, activation_).IsOK());
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    const auto* scale = context->Input<Tensor>(1);
    const auto* B = context->Input<Tensor>(2);
    const auto* mean = context->Input<Tensor>(3);
    const auto* var = context->Input<Tensor>(4);
    
    ORT_RETURN_IF_ERROR(BatchNormHelper::ValidateInputs(X, scale, B, mean, var, BatchNorm<T>::is_spatial_));

    const TensorShape& x_shape = X->Shape();
    Tensor* Y = context->Output(0, x_shape);

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
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
    
    // Saved mean corresponds to the mean from this batch
    // If these optional outputs are present (opset <= 9 or internal BN op) we re-use the space for calculations
    // Note that with opset <= 9 we will be outputting saved_inv_std_dev instead of saved_var
    Tensor *saved_mean = BatchNorm<T>::is_train_ ? context->Output(3, mean->Shape()) : nullptr;
    Tensor *saved_inv_std = BatchNorm<T>::is_train_ ? context->Output(4, var->Shape()) : nullptr;
    // With opset <= 9, both must be defined in training. If opset >= 14, neither should be defined in training
    ORT_ENFORCE(!BatchNorm<T>::is_train_ || ((!saved_mean && !saved_inv_std) || (saved_mean && saved_inv_std)), "Invalid number of outputs for BN training");
    Tensor saved_mean_allocated, saved_inv_std_allocated;
    if (BatchNorm<T>::is_train_ && !saved_mean) {
      saved_mean_allocated = Tensor(DataTypeImpl::GetType<T>(), mean->Shape(), alloc);
      saved_inv_std_allocated = Tensor(DataTypeImpl::GetType<T>(), var->Shape(), alloc);
      saved_mean = &saved_mean_allocated;
      saved_inv_std = &saved_inv_std_allocated;
    }
    ConstEigenArrayMap<T> X_arr(X->template Data<T>(),
                                BatchNorm<T>::is_spatial_ ? sample_size : sample_size_incl_all_channels,
                                BatchNorm<T>::is_spatial_ ? N * C : N);
    ConstEigenVectorArrayMap<T> scale_arr(scale->template Data<T>(), BatchNorm<T>::is_spatial_ ? C : sample_size_incl_all_channels);
    ConstEigenVectorArrayMap<T> bias_arr(B->template Data<T>(), BatchNorm<T>::is_spatial_ ? C : sample_size_incl_all_channels);

    // Note that we only support spatial BN for training
    if (BatchNorm<T>::is_train_) {
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
      auto* running_mean = context->Output(1, mean->Shape());
      auto* running_var = context->Output(2, var->Shape());
      const auto* input_running_mean = context->Input<Tensor>(3);
      const auto* input_running_var = context->Input<Tensor>(4);

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
      running_mean_arr = input_running_mean_arr * BatchNorm<T>::momentum_ + saved_mean_arr * (1. - BatchNorm<T>::momentum_);
      running_var_arr = input_running_var_arr * BatchNorm<T>::momentum_ + saved_var_arr * (1. - BatchNorm<T>::momentum_);
    }

    // Regardless of training or testing, we will apply the estimated mean
    // and standard deviation to the input. For testing, they are
    // specified directly by the input, and for training, they are computed
    // by the op.
    Eigen::Array<T, Eigen::Dynamic, 1> inv_std(BatchNorm<T>::is_spatial_ ? C : sample_size_incl_all_channels);

    if (!BatchNorm<T>::is_train_) {
      ConstEigenVectorArrayMap<T> var_arr(var->template Data<T>(), BatchNorm<T>::is_spatial_ ? C : sample_size_incl_all_channels);
      inv_std = (var_arr + BatchNorm<T>::epsilon_).sqrt().inverse();
    } else {
      EigenVectorArrayMap<T> saved_inv_std_arr(saved_inv_std->template MutableData<T>(), C);
      saved_inv_std_arr = (saved_inv_std_arr + BatchNorm<T>::epsilon_).inverse().sqrt();
      inv_std = saved_inv_std_arr;
    }

    // If we're training, do batch normalization based on computation from this batch
    ConstEigenVectorArrayMap<T> mean_arr(
        !BatchNorm<T>::is_train_ ? mean->template Data<T>() : saved_mean->template Data<T>(), BatchNorm<T>::is_spatial_ ? C : sample_size_incl_all_channels);

    // We can fuse the output computation as follows:
    //   ((x - est_mean) * (inv_var) * scale + bias
    // to
    //   (x * inv_var * scale) + (bias - est_mean * inv_var * scale)
    Eigen::Array<T, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
    Eigen::Array<T, Eigen::Dynamic, 1> new_bias = bias_arr - mean_arr * new_scale;
    EigenArrayMap<T> Y_arr(Y->template MutableData<T>(),
                           BatchNorm<T>::is_spatial_ ? sample_size : sample_size_incl_all_channels,
                           BatchNorm<T>::is_spatial_ ? N * C : N);

    if (BatchNorm<T>::is_spatial_) {  // spatial == 1
      for (size_t nc = 0; nc < N * C; ++nc) {
        Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
        Y_arr.col(nc) = Y_arr.col(nc).cwiseMax(0);
      }
    } else {  // spatial == 0
      for (size_t n = 0; n < N; ++n) {
        Y_arr.col(n) = X_arr.col(n) * new_scale.col(0) + new_bias.col(0);
        Y_arr.col(n) = Y_arr.col(n).cwiseMax(0);
      }
    }
    return Status::OK();
  }

 protected:
  MLAS_ACTIVATION activation_;
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedBatchNorm,
    kOnnxDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder().Alias(3,1).Alias(4,2).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedBatchNorm<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    FusedBatchNorm,
    kOnnxDomain,
    1,
    double,
    kCpuExecutionProvider,
    KernelDefBuilder().Alias(3,1).Alias(4,2).TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    FusedBatchNorm<double>);

// ONNX_CPU_OPERATOR_TYPED_KERNEL(FusedBatchNorm, 1, float,
//                                KernelDefBuilder().Alias(3,1).Alias(4,2).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
//                                FusedBatchNorm<float>);

// ONNX_CPU_OPERATOR_TYPED_KERNEL(FusedBatchNorm, 14, double,
//                                KernelDefBuilder().Alias(3,1).Alias(4,2).TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
//                                FusedBatchNorm<double>);


}  // namespace contrib
}  // namespace onnxruntime

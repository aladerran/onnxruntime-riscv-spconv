// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.



#include "systolic_element_wise_ops.h"
#include "core/providers/systolic/systolic_fwd.h"
#include "core/providers/systolic/helper/helper.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/common.h"
#include "core/providers/systolic/systolic_execution_provider.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace systolic {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    Add,
    kOnnxDomain,
    14,
    float,
    kSystolicExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Add<float>);

// ONNX_OPERATOR_TYPED_KERNEL_EX(
//     Add,
//     kOnnxDomain,
//     14,
//     double,
//     kSystolicExecutionProvider,
//     KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
//     Add<double>);

// ONNX_OPERATOR_TYPED_KERNEL_EX(
//     Add,
//     kOnnxDomain,
//     14,
//     int32_t,
//     kSystolicExecutionProvider,
//     KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
//     Add<int32_t>);
// ONNX_OPERATOR_TYPED_KERNEL_EX(
//     Add,
//     kOnnxDomain,
//     14,
//     int64_t,
//     kSystolicExecutionProvider,
//     KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int64_t>()),
//     Add<int64_t>);

namespace {
struct SystolicBroadcastHelper : public BroadcastHelper {
  SystolicBroadcastHelper(InputBroadcaster& input_broadcaster,
                         OutputBroadcaster& output_broadcaster,
                         ThreadPool* threadpool,
                         double unit_cost,
                         float A_scale_in, float B_scale_in, float C_scale_in,
                         char accelerator_mode)
      : BroadcastHelper{input_broadcaster, output_broadcaster, nullptr, threadpool, unit_cost},
        A_scale{A_scale_in},
        B_scale{B_scale_in},
        C_scale{C_scale_in},
        relu{false},
        accelerator_mode{accelerator_mode} {
  }

  SystolicBroadcastHelper(InputBroadcaster& input_broadcaster,
                         OutputBroadcaster& output_broadcaster,
                         ThreadPool* threadpool,
                         double unit_cost,
                         char accelerator_mode)
      : SystolicBroadcastHelper(input_broadcaster, output_broadcaster, threadpool, unit_cost,
                                1.0f, 1.0f, 1.0f, accelerator_mode) {
  }

  SystolicBroadcastHelper(const SystolicBroadcastHelper& rhs, size_t offset, size_t num_elements)
      : BroadcastHelper(rhs, offset, num_elements),
        A_scale{rhs.A_scale},
        B_scale{rhs.B_scale},
        C_scale{rhs.C_scale},
        relu{rhs.relu},
        accelerator_mode{rhs.accelerator_mode} {
  }

  float A_scale;
  float B_scale;
  float C_scale;
  bool relu;
  char accelerator_mode;
};

template <typename T>
void AddImpl(OpKernelContext& context, double unit_cost, const ProcessBroadcastSpanFuncs& functors, char acc_mode) {

  InputBroadcaster input_broadcaster{*context.Input<Tensor>(0), *context.Input<Tensor>(1)};
  OutputBroadcaster output_broadcaster{input_broadcaster.GetSpanSize(),
                                       *context.Output(0, input_broadcaster.GetOutputShape())};

  SystolicBroadcastHelper broadcast_helper(input_broadcaster, output_broadcaster,
                                          context.GetOperatorThreadPool(), unit_cost, acc_mode);

  BroadcastLooper(broadcast_helper, functors);
}

}  // namespace

template <typename T>
Status Add<T>::Compute(OpKernelContext* context) const {
  const ProcessBroadcastSpanFuncs functors = {
      [](BroadcastHelper& per_iter_bh) {
        // We don't yet support scalar + matrix resadd on systolic
        // We could do this via SW only by manually broadcasting
        // to systolic size and then mvin with 0 stride
        ORT_UNUSED_PARAMETER(per_iter_bh);
        ORT_NOT_IMPLEMENTED("Scalar + Matrix resadd on systolic not implemented");
        // SystolicBroadcastHelper& sbh = static_cast<SystolicBroadcastHelper&>(per_iter_bh);
        // const T input0 = per_iter_bh.ScalarInput0<T>();
        // auto input1 = per_iter_bh.SpanInput1<T>();
        // auto output = per_iter_bh.OutputSpan<T>();

        // SystolicAdd(
        //     sbh.accelerator_mode,
        //     sbh.relu,
        //     input0, sbh.A_scale,
        //     input1.data(), sbh.B_scale,
        //     output.data(), sbh.C_scale,
        //     output.size());
      },
      [](BroadcastHelper& per_iter_bh) {
        ORT_UNUSED_PARAMETER(per_iter_bh);
        ORT_UNUSED_PARAMETER(per_iter_bh);
        // SystolicBroadcastHelper& sbh = static_cast<SystolicBroadcastHelper&>(per_iter_bh);
        // auto input0 = per_iter_bh.SpanInput0<T>();
        // const T input1 = per_iter_bh.ScalarInput1<T>();
        // auto output = per_iter_bh.OutputSpan<T>();
        // SystolicAdd(
        //     sbh.accelerator_mode,
        //     sbh.relu,
        //     input0.data(), sbh.A_scale,
        //     input1, sbh.B_scale,
        //     output.data(), sbh.C_scale,
        //     output.size());
      },
      [](BroadcastHelper& per_iter_bh) {
        SystolicBroadcastHelper& sbh = static_cast<SystolicBroadcastHelper&>(per_iter_bh);
        auto input0 = per_iter_bh.SpanInput0<T>();
        auto input1 = per_iter_bh.SpanInput1<T>();
        auto output = per_iter_bh.OutputSpan<T>();

        SystolicAdd(
            sbh.accelerator_mode,
            sbh.relu,
            input0.data(), sbh.A_scale,
            input1.data(), sbh.B_scale,
            output.data(), sbh.C_scale,
            output.size()
            );
      }};
  AddImpl<T>(*context, 1.0, functors, 
    static_cast<const SystolicExecutionProvider*>(this->Info().GetExecutionProvider())->GetAcceleratorMode());
  return Status::OK();
}

// template <typename T>
// Status Sub<T>::Compute(OpKernelContext* context) const{}
} // namespace systolic
}  // namespace onnxruntime

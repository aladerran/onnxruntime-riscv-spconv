#pragma once

#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace systolic {

template <typename T>
class SystolicAddRelu final : public OpKernel {
 public:
  SystolicAddRelu(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

} // namespace systolic
}  // namespace onnxruntime
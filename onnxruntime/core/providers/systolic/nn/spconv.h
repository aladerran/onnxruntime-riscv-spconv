#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/systolic/nn/spconv_attributes.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "pool_attributes.h"

namespace onnxruntime {
namespace systolic {

template <typename T>
class SpConv3d : public OpKernel {
 public:
  SpConv3d(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  SpConvAttributes conv_attrs_;
  bool fused_relu_ = false;
};

} // namespace systolic
}  // namespace onnxruntime

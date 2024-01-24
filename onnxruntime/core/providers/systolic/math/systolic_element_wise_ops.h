// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace systolic {

template <typename T>
class Add final : public OpKernel {
 public:
  Add(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;
};

// template <typename T>
// class Sub final : public OpKernel {
//  public:
//   Sub(const OpKernelInfo& info) : OpKernel(info) {
//   }

//   Status Compute(OpKernelContext* context) const override;
// };
} // namespace systolic
}  // namespace onnxruntime

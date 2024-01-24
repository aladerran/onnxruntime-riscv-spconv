#pragma once

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include "pool_attributes.h"
#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/providers/common.h"


namespace onnxruntime {
namespace systolic {

template <typename T>
class BatchNorm : public OpKernel {
 public:
  explicit BatchNorm(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info),
                                                           is_spatial_(op_kernel_info.GetAttrOrDefault<int64_t>("spatial", 1) == 1) {
    auto st = op_kernel_info.GetAttr<float>("epsilon", &epsilon_);
    ORT_ENFORCE(st.IsOK(), st.ErrorMessage());
    auto mt = op_kernel_info.GetAttr<float>("momentum", &momentum_);
    ORT_ENFORCE(mt.IsOK(), mt.ErrorMessage());
    // For opset 6-8, if spatial attribute exists, pick up the value (by default spatial == 1)
    // From opset 9 onwards, by default, only the spatial case (spatial == 1) is defined per spec

    // For opset 14 onwards, training is true iff we have optional outputs present
    // For opset < 14, since no training attribute is present we assume optional outputs indicate training mode 
    is_train_ = OpKernel::Node().OutputDefs().size() > 1;
    ORT_ENFORCE(!is_train_ || is_spatial_, "Training mode does not support non-spatial BN");
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

 protected:
  float epsilon_;
  float momentum_;
  const bool is_spatial_;
  int64_t is_train_;
};

} // namespace systolic
}  // namespace onnxruntime
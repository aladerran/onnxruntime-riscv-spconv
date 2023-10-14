#pragma once

#include "core/common/common.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/providers/common.h"
#include "core/util/math.h"

namespace onnxruntime {

// A helper struct holding attributes for Conv-family ops
struct SpConvAttributes {
  explicit SpConvAttributes(const OpNodeProtoHelper<ProtoHelperNodeContext>& info) {

    kernel_shape_specified = info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK();

    auto status = info.GetAttrs<int64_t>("strides", strides);
    if (!status.IsOK() || strides.empty()) {
      strides.resize(kernel_shape_.size(), 1);
    }

    status = info.GetAttrs<int64_t>("dilations", dilations);
    if (!status.IsOK() || dilations.empty()) {
      dilations.resize(kernel_shape_.size(), 1);
    }

    status = info.GetAttr<int64_t>("transposed", &transposed); 
    if (!status.IsOK() || dilations.empty()) {
      transposed = 0 ;
    }

  }

  ~SpConvAttributes() = default;

  Status ComputeSpKernelShape(const TensorShape& weight_shape, std::vector<int64_t>& kernel_shape) const {
    if (kernel_shape_specified) {
      kernel_shape = kernel_shape_;
      if ( kernel_shape.size() != 3 ){
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape gets wrong num_dims, should be [3].",
                               " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str());
      }
      int64_t kernel_volume = 1;
      for (size_t i =0; i < 3 ; i++){
        kernel_volume *= kernel_shape[i];
      }
      if ( kernel_volume != weight_shape[0]){
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel volume is not compatible with Weight dim0, "\
                               "Weigh.shape[0] should be the production of kernel_shape.",
                               " kernel_volume: ", std::to_string(kernel_volume),
                               " Weight shape: ", weight_shape.ToString().c_str());
      }
    } else {
      auto& weight_dims = weight_shape.GetDims();
      kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
    }

    return Status::OK();
  }

  Status ValidateSpInputShape(const  TensorShape& coords_shape, const  TensorShape& feats_shape, 
                            const  TensorShape& strides_shape, const  TensorShape& weight_shape) const {
    if (coords_shape.NumDimensions() !=2 || coords_shape[1] != 4 ){
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "InputCoords expects a N x 4 Tensor which has 2 dims, ",
                             " got InputCoords shape: ", coords_shape.ToString().c_str());
    }
    if (feats_shape.NumDimensions() !=2 || feats_shape[1] != 4 ){
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "InputFeats expects a N x 4 Tensor which has 2 dims, ",
                             " got InputFeats shape: ", feats_shape.ToString().c_str());
    }
    if (strides_shape.NumDimensions() !=1 || strides_shape[0] != 3 ){
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "InputStrides expects a 1 x 3 Tensor , ",
                             " got InputStrides shape: ", strides_shape.ToString().c_str());
    }
    if (weight_shape.NumDimensions() !=3 ){
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Weight expects a 3 dims Tensor , ",
                             " got shape: ", weight_shape.ToString().c_str());
    }
    if ( coords_shape[0] != feats_shape[0] ){
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "the numbur of coordinates in InputCoords does not match " \
                             "the number of features in InputFeats, they should be equal.",
                             " InputCoords: ", coords_shape.ToString().c_str(),
                             " InputFeats: ", weight_shape.ToString().c_str());
    }
    if( feats_shape[1] != weight_shape[1] ){
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "width(in channels) of InputFeats does not match " \
                             "that of Weight.",
                             " InputCoords: ", feats_shape.ToString().c_str(),
                             " Weight: ", weight_shape.ToString().c_str());
    }
    // ignore checking weight[0], the 'kernel volume', it will be done in the above function ComputeKernelShape
    // ignore checking weight[2], also the output channels, as sizes_io is an optional input
    return Status::OK();
  }

  Status ValidateInputShape(const Tensor* coords, const Tensor* feats, 
                            const Tensor* strides, const Tensor* weight) const {
    return ValidateInputShape(coords->Shape(), feats->Shape(), 
                              strides->Shape(), weight->Shape());
  }

/*
  Status InferOutputShape(const TensorShape& input_shape,
                          const std::vector<int64_t>& kernel_shape,
                          const std::vector<int64_t>& strides_p,
                          const std::vector<int64_t>& dilations_p,
                          std::vector<int64_t>& pads_p,
                          std::vector<int64_t>& output_shape,
                          bool force_symmetric_auto_padding = false) const {
    size_t rank = input_shape.NumDimensions();

    // Make sure all "metadata" containers have the right number of elements
    if (rank > strides_p.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in strides. Expected: ", rank, " Got: ", strides_p.size());

    if (rank > kernel_shape.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in kernel shape. Expected: ", rank, " Got: ", kernel_shape.size());

    if (rank > dilations_p.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in dilations. Expected: ", rank, " Got: ", dilations_p.size());

    if ((2 * rank) > pads_p.size())
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Not enough elements in pads. Expected: ", (2 * rank), " Got: ", pads_p.size());

    for (size_t dim = 0; dim < rank; ++dim) {
      int64_t output_dim_size = 0;
      ORT_RETURN_IF_ERROR(ComputePadAndOutputShape(input_shape[dim],
                                                   strides_p[dim],
                                                   kernel_shape[dim],
                                                   dilations_p[dim],
                                                   auto_pad,
                                                   pads_p.at(dim),
                                                   pads_p.at(input_shape.NumDimensions() + dim),
                                                   output_dim_size,
                                                   force_symmetric_auto_padding));
      if (output_dim_size <= 0) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid input shape: " + input_shape.ToString());
      }
      output_shape.push_back(output_dim_size);
    }
    return Status::OK();
  }

*/


  bool kernel_shape_specified;
//   std::vector<int64_t> kernel_shape;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  int64_t transposed;

 private:
  std::vector<int64_t> kernel_shape_;  // must use ComputeKernelShape(...), instead of kernel_shape_
};

}  // namespace onnxruntime
/*!
 * Copyright (c) 2016 by Contributors
 * \file grad_scale.cc
 * \brief 
 * \author Ye Yuan
*/
#include "./grad_scale-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(GradScaleParam param) {
  return new GradScaleOp<cpu>(param);
}

Operator *GradScaleProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

// DO_BIND_DISPATCH comes from operator_common.h
/*
Operator *GradScaleProp::CreateOperatorEx(Context ctx, 
                                          std::vector<TShape> *in_shape,
                                          std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}
*/

DMLC_REGISTER_PARAMETER(GradScaleParam);

MXNET_REGISTER_OP_PROPERTY(GradScale, GradScaleProp)
.describe("Scale gradient in this layer.")
.add_argument("data", "Symbol", "Input data")
.add_argument("scale", "Symbol", "Scale gradient")
.add_arguments(GradScaleParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

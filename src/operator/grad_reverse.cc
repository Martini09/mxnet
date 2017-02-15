/*!
 * Copyright (c) 2016 by Contributors
 * \file grad_reverse.cc
 * \brief 
 * \author Ye Yuan
*/
#include "./grad_reverse-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(GradReverseParam param) {
  return new GradReverseOp<cpu>(param);
}

Operator *GradReverseProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(GradReverseParam);

MXNET_REGISTER_OP_PROPERTY(GradReverse, GradReverseProp)
.describe("Reverse gradient in this layer.")
.add_argument("data", "Symbol", "Reverse gradient.")
.add_arguments(GradReverseParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

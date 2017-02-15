/*!
 * Copyright (c) 2016 by Contributors
 * \file grad_reverse.cu
 * \brief
 * \author Ye Yuan
*/

#include "./grad_reverse-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(GradReverseParam param) {
  return new GradReverseOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

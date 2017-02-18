/*!
 * Copyright (c) 2016 by Contributors
 * \file grad_scale.cu
 * \brief
 * \author Ye Yuan
*/

#include "./grad_scale-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(GradScaleParam param) {
  return new GradScaleOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet

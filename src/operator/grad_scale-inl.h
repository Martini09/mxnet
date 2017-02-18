/*!
 * Copyright (c) 2016 by Contributors
 * \file grad_scale-inl.h
 * \brief Scale the gradient
 * \author Ye Yuan
*/

#ifndef MXNET_OPERATOR_GRAD_SCALE_INL_H
#define MXNET_OPERATOR_GRAD_SCALE_INL_H

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

#include <iostream>
#include <fstream>
using namespace std;

namespace mxnet {
namespace op {

namespace grad_scale {
enum GradScaleOpInputs {kData, kScale};
enum GradScaleOpOutputs {kOut};
} // namespcae grad_scale

/* Parameters for layer */
struct GradScaleParam: public dmlc::Parameter<GradScaleParam> {
    float base_scale;
    DMLC_DECLARE_PARAMETER(GradScaleParam) {
        DMLC_DECLARE_FIELD(base_scale).set_default(1.0f)
        .describe("Scale the gradient by a float factor");
    };
};

/* Forward and Backward */
template<typename xpu>
class GradScaleOp: public Operator {
public:
  explicit GradScaleOp(GradScaleParam param): param_(param) {}

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {

    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2) << "Gradient Scale Input: [data, scale]";
    CHECK_EQ(out_data.size(), 1) << "Gradient Scale Output: [output]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> data = in_data[grad_scale::kData].FlatTo2D<xpu, real_t>(s);
    Tensor<xpu, 2> out = out_data[grad_scale::kOut].FlatTo2D<xpu, real_t>(s);
    /* Pass input data */
    out = F<mshadow_op::identity>(data);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {

    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1> gdata = in_grad[grad_scale::kData].FlatTo1D<xpu, real_t>(s);    
    Tensor<xpu, 1> grad = out_grad[grad_scale::kOut].FlatTo1D<xpu, real_t>(s);    
    Tensor<xpu, 1> scale = in_data[grad_scale::kScale].FlatTo1D<xpu, real_t>(s);

    /* Scale gradient */
    gdata = param_.base_scale * grad * scale;
  }

private:
  GradScaleParam param_;

}; // class GradScaleOp


/* Decalre Factory function, used for dispatch specialization */
template<typename xpu>
Operator* CreateOp(GradScaleParam param);

#if DMLC_USE_CXX11
class GradScaleProp: public OperatorProperty {
public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "scale"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input: [data, scale]";
    const TShape &dshape = (*in_shape)[grad_scale::kData];
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new GradScaleProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "GradScale";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[grad_scale::kScale], out_grad[grad_scale::kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[grad_scale::kOut], in_grad[grad_scale::kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[grad_scale::kData], out_data[grad_scale::kOut]}};
  }

  Operator* CreateOperator(Context ctx) const;

private:
  GradScaleParam param_;
};         // class GradScaleProp
#endif     // DMLC_USE_CXX11


} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_GRAD_SCALE_INL_H 

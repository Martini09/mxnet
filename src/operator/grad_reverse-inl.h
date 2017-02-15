/*!
 * Copyright (c) 2016 by Contributors
 * \file grad_reverse-inl.h
 * \brief Reverse gradient
 * \author Ye Yuan
*/

#ifndef MXNET_OPERATOR_GRAD_REVERSE_INL_H
#define MXNET_OPERATOR_GRAD_REVERSE_INL_H

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

/* Get negative of a value */
struct negative {
	MSHADOW_XINLINE static real_t Map(real_t a) {
		return -a;
	}
};

enum GradReverseOpInputs {kData};
enum GradReverseOpOutputs {kOut};

/* Parameters for layer */
struct GradReverseParam: public dmlc::Parameter<GradReverseParam> {
	float grad_scale;
	DMLC_DECLARE_PARAMETER(GradReverseParam) {
    	DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    	.describe("Scale the reverse gradient by a float factor");
  	};
};

/* Forward and Backward */
template<typename xpu>
class GradReverseOp: public Operator {
public:
	explicit GradReverseOp(GradReverseParam param): param_(param) {}

	virtual void Forward(const OpContext &ctx,
						 					 const std::vector<TBlob> &in_data,
     					         const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {

		using namespace mshadow;
		using namespace mshadow::expr;
		CHECK_EQ(in_data.size(), 1) << "Gradient Reverse Input: [data]";
		CHECK_EQ(out_data.size(), 1) << "Gradient Reverse Output: [output]";
		Stream<xpu> *s = ctx.get_stream<xpu>();
		Tensor<xpu, 2> data = in_data[kData].FlatTo2D<xpu, real_t>(s);
		Tensor<xpu, 2> out = out_data[kOut].FlatTo2D<xpu, real_t>(s);
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
    CHECK_EQ(in_grad.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2> gdata = in_grad[kData].FlatTo2D<xpu, real_t>(s);    
    Tensor<xpu, 2> grad = out_grad[kOut].FlatTo2D<xpu, real_t>(s);    
    /* Reverse gradient */
    gdata = param_.grad_scale * F<mshadow_op::negation>(grad); 
	}

private:
	GradReverseParam param_;

}; // class GradReverseOp


/* Decalre Factory function, used for dispatch specialization */
template<typename xpu>
Operator* CreateOp(GradReverseParam param);

#if DMLC_USE_CXX11
class GradReverseProp: public OperatorProperty {
public:
  std::vector<std::string> ListArguments() const override {
    return {"data"};
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
    CHECK_EQ(in_shape->size(), 1) << "Input: [data]";
    const TShape &dshape = (*in_shape)[kData];
    if (dshape.ndim() == 0) return false;
    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new GradReverseProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "GradReverse";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[kOut]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[kOut], in_grad[kData]}};
  }

  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[kData], out_data[kOut]}};
  }

  Operator* CreateOperator(Context ctx) const;

private:
  GradReverseParam param_;
};         // class GradReverseProp
#endif     // DMLC_USE_CXX11


} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_GRAD_REVERSE_INL_H 

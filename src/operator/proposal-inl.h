/*!
 * Copyright (c) 2015 by Contributors
 * \file proposal-inl.h
 * \brief Proposal Operator
 * \author Piotr Teterwak, Jian Guo
*/
#ifndef MXNET_OPERATOR_PROPOSAL_INL_H_
#define MXNET_OPERATOR_PROPOSAL_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./native_op-inl.h"
#include "./rcnn_utils.h"


// extend NumericalParam
namespace mxnet {
namespace op {

/*!
* \brief structure for numerical tuple input
* \tparam VType data type of param
*/
template<typename VType>
struct NumericalParam {
  NumericalParam() {}
  explicit NumericalParam(VType *begin, VType *end) {
    int32_t size = static_cast<int32_t>(end - begin);
    info.resize(size);
    for (int i = 0; i < size; ++i) {
      info[i] = *(begin + i);
    }
  }
  inline size_t ndim() const {
    return info.size();
  }
  std::vector<VType> info;
};

template<typename VType>
inline std::istream &operator>>(std::istream &is, NumericalParam<VType> &param) {
  while (true) {
    char ch = is.get();
    if (ch == '(') break;
    if (!isspace(ch)) {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  VType idx;
  std::vector<VType> tmp;
  // deal with empty case
  size_t pos = is.tellg();
  char ch = is.get();
  if (ch == ')') {
    param.info = tmp;
    return is;
  }
  is.seekg(pos);
  // finish deal
  while (is >> idx) {
    tmp.push_back(idx);
    char ch;
    do {
      ch = is.get();
    } while (isspace(ch));
    if (ch == ',') {
      while (true) {
        ch = is.peek();
        if (isspace(ch)) {
          is.get(); continue;
        }
        if (ch == ')') {
          is.get(); break;
        }
        break;
      }
      if (ch == ')') break;
    } else if (ch == ')') {
      break;
    } else {
      is.setstate(std::ios::failbit);
      return is;
    }
  }
  param.info = tmp;
  return is;
}

template<typename VType>
inline std::ostream &operator<<(std::ostream &os, const NumericalParam<VType> &param) {
  os << '(';
  for (index_t i = 0; i < param.info.size(); ++i) {
    if (i != 0) os << ',';
    os << param.info[i];
  }
  // python style tuple
  if (param.info.size() == 1) os << ',';
  os << ')';
  return os;
}

}
}

namespace mxnet {
namespace op {

namespace proposal {
enum ProposalOpInputs {kClsProb, kBBoxPred, kImInfo};
enum ProposalOpOutputs {kOut, kScore};
enum ProposalForwardResource {kTempResource};
}  // proposal


struct ProposalParam : public dmlc::Parameter<ProposalParam> {
  int rpn_pre_nms_top_n;
  int rpn_post_nms_top_n;
  float threshold;
  int rpn_min_size;
  NumericalParam<float> scales;
  NumericalParam<float> ratios;
  int feature_stride;
  bool output_score;
  bool iou_loss;
  DMLC_DECLARE_PARAMETER(ProposalParam) {
    float tmp[] = {0, 0, 0, 0};
    DMLC_DECLARE_FIELD(rpn_pre_nms_top_n).set_default(6000)
    .describe("Number of top scoring boxes to keep after applying NMS to RPN proposals");
    DMLC_DECLARE_FIELD(rpn_post_nms_top_n).set_default(300)
    .describe("Overlap threshold used for non-maximum"
              "suppresion(suppress boxes with IoU >= this threshold");
    DMLC_DECLARE_FIELD(threshold).set_default(0.7)
    .describe("NMS value, below which to suppress.");
    DMLC_DECLARE_FIELD(rpn_min_size).set_default(16)
    .describe("Minimum height or width in proposal");
    tmp[0] = 4.0f; tmp[1] = 8.0f; tmp[2] = 16.0f; tmp[3] = 32.0f;
    DMLC_DECLARE_FIELD(scales).set_default(NumericalParam<float>(tmp, tmp + 4))
    .describe("Used to generate anchor windows by enumerating scales");
    tmp[0] = 0.5f; tmp[1] = 1.0f; tmp[2] = 2.0f;
    DMLC_DECLARE_FIELD(ratios).set_default(NumericalParam<float>(tmp, tmp + 3))
    .describe("Used to generate anchor windows by enumerating ratios");
    DMLC_DECLARE_FIELD(feature_stride).set_default(16)
    .describe("The size of the receptive field each unit in the convolution layer of the rpn,"
              "for example the product of all stride's prior to this layer.");
    DMLC_DECLARE_FIELD(output_score).set_default(false)
    .describe("Add score to outputs");
    DMLC_DECLARE_FIELD(iou_loss).set_default(false)
    .describe("Usage of IoU Loss");
  }
};

template<typename xpu>
class ProposalOp : public Operator{
 public:
  explicit ProposalOp(ProposalParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 2);
    CHECK_GT(req.size(), 1);
    CHECK_EQ(req[proposal::kOut], kWriteTo);
    CHECK_EQ(in_data[proposal::kClsProb].shape_[0], 1) << "Sorry, multiple images each device is not implemented.";

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Shape<4> scores_shape = Shape4(in_data[proposal::kClsProb].shape_[0],
                                   in_data[proposal::kClsProb].shape_[1] / 2,
                                   in_data[proposal::kClsProb].shape_[2],
                                   in_data[proposal::kClsProb].shape_[3]);
    real_t* foreground_score_ptr = reinterpret_cast<real_t *>(in_data[proposal::kClsProb].dptr_) + scores_shape.Size();
    Tensor<cpu, 4> scores = Tensor<cpu, 4>(foreground_score_ptr, scores_shape);
    Tensor<cpu, 4> bbox_deltas = in_data[proposal::kBBoxPred].get<cpu, 4, real_t>(s);
    Tensor<cpu, 2> im_info = in_data[proposal::kImInfo].get<cpu, 2, real_t>(s);

    Tensor<cpu, 2> out = out_data[proposal::kOut].get<cpu, 2, real_t>(s);
    Tensor<cpu, 2> out_score = out_data[proposal::kScore].get<cpu, 2, real_t>(s);

    int num_anchors = in_data[proposal::kClsProb].shape_[1] / 2;
    int height = scores.size(2);
    int width = scores.size(3);
    int count = num_anchors * height * width;
    int rpn_pre_nms_top_n = (param_.rpn_pre_nms_top_n > 0) ? param_.rpn_pre_nms_top_n : count;
    rpn_pre_nms_top_n = std::min(rpn_pre_nms_top_n, count);
    int rpn_post_nms_top_n = std::min(param_.rpn_post_nms_top_n, rpn_pre_nms_top_n);

    Tensor<cpu, 2> workspace_proposals = ctx.requested[proposal::kTempResource].get_space<cpu>(
      Shape2(count, 5), s);
    Tensor<cpu, 2> workspace_ordered_proposals = ctx.requested[proposal::kTempResource].get_space<cpu>(
      Shape2(rpn_pre_nms_top_n, 5), s);
    Tensor<cpu, 2> workspace_pre_nms = ctx.requested[proposal::kTempResource].get_space<cpu>(
      Shape2(2, count), s);
    Tensor<cpu, 2> workspace_nms = ctx.requested[proposal::kTempResource].get_space<cpu>(
      Shape2(3, rpn_pre_nms_top_n), s);

    // Generate anchors
    std::vector<float> base_anchor(4);
    base_anchor[0] = 0.0;
    base_anchor[1] = 0.0;
    base_anchor[2] = param_.feature_stride - 1.0;
    base_anchor[3] = param_.feature_stride - 1.0;
    CHECK_EQ(num_anchors, param_.ratios.info.size() * param_.scales.info.size());
    std::vector<float> anchors;
    utils::GenerateAnchors(base_anchor,
                           param_.ratios.info,
                           param_.scales.info,
                           anchors);
    std::memcpy(workspace_proposals.dptr_, &anchors[0], sizeof(float) * anchors.size());

    //Enumerate all shifted anchors
    for (index_t i = 0; i < num_anchors; ++i){
      for (index_t j = 0; j < height; ++j){
        for (index_t k = 0; k < width; ++k){
          index_t index = j * (width * num_anchors) + k * (num_anchors) + i;
          workspace_proposals[index][0] = workspace_proposals[i][0] + k * param_.feature_stride;
          workspace_proposals[index][1] = workspace_proposals[i][1] + j * param_.feature_stride;
          workspace_proposals[index][2] = workspace_proposals[i][2] + k * param_.feature_stride;
          workspace_proposals[index][3] = workspace_proposals[i][3] + j * param_.feature_stride;
          workspace_proposals[index][4] = scores[0][i][j][k];
        }
      }
    }

    // prevent padded predictions
    int real_height = static_cast<int>(im_info[0][0] / param_.feature_stride);
    int real_width = static_cast<int>(im_info[0][1] / param_.feature_stride);
    CHECK_GE(height, real_height) << height << " " << real_height << std::endl;
    CHECK_GE(width, real_width) << width << " " << real_width << std::endl;

    if (param_.iou_loss) {
      utils::IoUTransformInv(workspace_proposals, bbox_deltas, im_info[0][0], im_info[0][1],
                             real_height, real_width, &(workspace_proposals));
    } else {
      utils::BBoxTransformInv(workspace_proposals, bbox_deltas, im_info[0][0], im_info[0][1],
                              real_height, real_width, &(workspace_proposals));
    }
    utils::FilterBox(workspace_proposals, param_.rpn_min_size * im_info[0][2]);

    Tensor<cpu, 1> score = workspace_pre_nms[0];
    Tensor<cpu, 1> order = workspace_pre_nms[1];

    utils::CopyScore(workspace_proposals,
                     score,
                     order);
    utils::ReverseArgsort(score,
                          order);
    utils::ReorderProposals(workspace_proposals,
                            order,
                            rpn_pre_nms_top_n,
                            workspace_ordered_proposals);

    index_t out_size = 0;
    Tensor<cpu, 1> area = workspace_nms[0];
    Tensor<cpu, 1> suppressed = workspace_nms[1];
    Tensor<cpu, 1> keep = workspace_nms[2];

    utils::NonMaximumSuppression(workspace_ordered_proposals,
                                 param_.threshold,
                                 rpn_post_nms_top_n,
                                 area,
                                 suppressed,
                                 keep,
                                 &out_size);

    // fill in output rois
    for (index_t i = 0; i < out.size(0); ++i) {
      //batch index 0
      out[i][0] = 0;
      if (i < out_size) {
        index_t index = keep[i];
        for (index_t j = 0; j < 4; ++j) {
          out[i][j + 1] =  workspace_ordered_proposals[index][j];
        }
      } else {
        index_t index = keep[i % out_size];
        for (index_t j = 0; j < 4; ++j) {
          out[i][j + 1] = workspace_ordered_proposals[index][j];
        }
      }
    }

    // fill in output score
    for (index_t i = 0; i < out_score.size(0); i++) {
      if (i < out_size) {
        index_t index = keep[i];
        out_score[i][0] = workspace_ordered_proposals[index][4];
      }
      else {
        index_t index = keep[i % out_size];
        out_score[i][0] = workspace_ordered_proposals[index][4];
      }
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 3);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> gscores = in_grad[proposal::kClsProb].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> gbbox = in_grad[proposal::kBBoxPred].get<xpu, 4, real_t>(s);
    Tensor<xpu, 2> ginfo = in_grad[proposal::kImInfo].get<xpu, 2, real_t>(s);

    // can not assume the grad would be zero
    Assign(gscores, req[proposal::kClsProb], 0);
    Assign(gbbox, req[proposal::kBBoxPred], 0);
    Assign(ginfo, req[proposal::kImInfo], 0);
  }

 private:
  ProposalParam param_;
};  // class ProposalOp

template<typename xpu>
Operator *CreateOp(ProposalParam param);


#if DMLC_USE_CXX11
class ProposalProp : public OperatorProperty {
 public:
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
    CHECK_EQ(in_shape->size(), 3) << "Input:[cls_prob, bbox_pred, im_info]";
    const TShape &dshape = in_shape->at(proposal::kClsProb);
    if (dshape.ndim() == 0) return false;
    Shape<4> bbox_pred_shape;
    bbox_pred_shape = Shape4(dshape[0], dshape[1] * 2, dshape[2], dshape[3]);
    SHAPE_ASSIGN_CHECK(*in_shape, proposal::kBBoxPred,
                       bbox_pred_shape);
    Shape<2> im_info_shape;
    im_info_shape = Shape2(dshape[0], 3);
    SHAPE_ASSIGN_CHECK(*in_shape, proposal::kImInfo, im_info_shape);
    out_shape->clear();
    // output
    out_shape->push_back(Shape2(param_.rpn_post_nms_top_n, 5));
    // score
    out_shape->push_back(Shape2(param_.rpn_post_nms_top_n, 1));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ProposalProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "Proposal";
  }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  int NumVisibleOutputs() const override {
    if (param_.output_score) {
      return 2;
    }
    else{
      return 1;
    }
  }

  int NumOutputs() const override {
    return 2;
  }

  std::vector<std::string> ListArguments() const override {
    return {"cls_prob", "bbox_pred", "im_info"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "score"};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  ProposalParam param_;
};  // class ProposalProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  //  MXNET_OPERATOR_PROPOSAL_INL_H_

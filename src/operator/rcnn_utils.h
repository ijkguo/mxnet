/*!
 * Copyright (c) 2015 by Contributors
 * \file rcnn_utils.h
 * \brief Proposal Operator
 * \author Bing Xu, Jian Guo
*/
#ifndef MXNET_OPERATOR_RCNN_UTILS_H_
#define MXNET_OPERATOR_RCNN_UTILS_H_
#include <algorithm>
#include <mshadow/tensor.h>
#include <mshadow/extension.h>
#include <iostream>

//========================
// Anchor Generation Utils
//========================
namespace mxnet {
namespace op {
namespace utils {

inline void _MakeAnchor(float w,
                        float h,
                        float x_ctr,
                        float y_ctr,
                        std::vector<float>& out_anchors) {
  out_anchors.push_back(x_ctr - 0.5f * (w - 1.0f));
  out_anchors.push_back(y_ctr - 0.5f * (h - 1.0f));
  out_anchors.push_back(x_ctr + 0.5f * (w - 1.0f));
  out_anchors.push_back(y_ctr + 0.5f * (h - 1.0f));
  out_anchors.push_back(0.0f);
}

inline void _Transform(float scale,
                       float ratio,
                       const std::vector<float>& base_anchor,
                       std::vector<float>& out_anchors) {
  float w = base_anchor[2] - base_anchor[1] + 1.0f;
  float h = base_anchor[3] - base_anchor[1] + 1.0f;
  float x_ctr = base_anchor[0] + 0.5 * (w - 1.0f);
  float y_ctr = base_anchor[1] + 0.5 * (h - 1.0f);
  float size = w * h;
  float size_ratios = std::floor(size / ratio);
  float new_w = std::floor(std::sqrt(size_ratios) + 0.5f) * scale;
  float new_h = std::floor((new_w / scale * ratio) + 0.5f) * scale;

  _MakeAnchor(new_w, new_h, x_ctr,
             y_ctr, out_anchors);
}

// out_anchors must have shape (n, 5), where n is ratios.size() * scales.size()
inline void GenerateAnchors(const std::vector<float>& base_anchor,
                            const std::vector<float>& ratios,
                            const std::vector<float>& scales,
                            std::vector<float>& out_anchors) {
  for (size_t j = 0; j < ratios.size(); ++j) {
    for (size_t k = 0; k < scales.size(); ++k) {
      _Transform(scales[k], ratios[j], base_anchor, out_anchors);
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

//============================
// Bounding Box Transform Utils
//============================
namespace mxnet {
namespace op {
namespace utils {

// bbox prediction and clip to the image borders
inline void BBoxTransformInv(const mshadow::Tensor<cpu, 2>& boxes,
                             const mshadow::Tensor<cpu, 4>& deltas,
                             const float im_height,
                             const float im_width,
                             mshadow::Tensor<cpu, 2> *out_pred_boxes) {
  CHECK_GE(boxes.size(1), 4);
  CHECK_GE(out_pred_boxes->size(1), 4);
  size_t anchors = deltas.size(1)/4;
  size_t heights = deltas.size(2);
  size_t widths = deltas.size(3);

  for (size_t a = 0; a < anchors; ++a) {
    for (size_t h = 0; h < heights; ++h) {
      for (size_t w = 0; w < widths; ++w) {
        index_t index = h * (widths * anchors) + w * (anchors) + a;
        float width = boxes[index][2] - boxes[index][0] + 1.0;
        float height = boxes[index][3] - boxes[index][1] + 1.0;
        float ctr_x = boxes[index][0] + 0.5 * width;
        float ctr_y = boxes[index][1] + 0.5 * height;

        float dx = deltas[0][a*4 + 0][h][w];
        float dy = deltas[0][a*4 + 1][h][w];
        float dw = deltas[0][a*4 + 2][h][w];
        float dh = deltas[0][a*4 + 3][h][w];

        float pred_ctr_x = dx * width + ctr_x;
        float pred_ctr_y = dy * height + ctr_y;
        float pred_w = exp(dw) * width;
        float pred_h = exp(dh) * height;

        float pred_x1 = pred_ctr_x - 0.5 * pred_w;
        float pred_y1 = pred_ctr_y - 0.5 * pred_h;
        float pred_x2 = pred_ctr_x + 0.5 * pred_w;
        float pred_y2 = pred_ctr_y + 0.5 * pred_h;

        pred_x1 = std::max(std::min(pred_x1, im_width - 1.0f), 0.0f);
        pred_y1 = std::max(std::min(pred_y1, im_height - 1.0f), 0.0f);
        pred_x2 = std::max(std::min(pred_x2, im_width - 1.0f), 0.0f);
        pred_y2 = std::max(std::min(pred_y2, im_height - 1.0f), 0.0f);

        (*out_pred_boxes)[index][0] = pred_x1;
        (*out_pred_boxes)[index][1] = pred_y1;
        (*out_pred_boxes)[index][2] = pred_x2;
        (*out_pred_boxes)[index][3] = pred_y2;
      }
    }
  }
}

// iou prediction and clip to the image border
inline void IoUTransformInv(const mshadow::Tensor<cpu, 2>& boxes,
                            const mshadow::Tensor<cpu, 4>& deltas,
                            const float im_height,
                            const float im_width,
                            mshadow::Tensor<cpu, 2> *out_pred_boxes) {
  CHECK_GE(boxes.size(1), 4);
  CHECK_GE(out_pred_boxes->size(1), 4);
  size_t anchors = deltas.size(1)/4;
  size_t heights = deltas.size(2);
  size_t widths = deltas.size(3);

  for (size_t a = 0; a < anchors; ++a) {
    for (size_t h = 0; h < heights; ++h) {
      for (size_t w = 0; w < widths; ++w) {
        index_t index = h * (widths * anchors) + w * (anchors) + a;
        float x1 = boxes[index][0];
        float y1 = boxes[index][1];
        float x2 = boxes[index][2];
        float y2 = boxes[index][3];

        float dx1 = deltas[0][a * 4 + 0][h][w];
        float dy1 = deltas[0][a * 4 + 1][h][w];
        float dx2 = deltas[0][a * 4 + 2][h][w];
        float dy2 = deltas[0][a * 4 + 3][h][w];

        float pred_x1 = x1 + dx1;
        float pred_y1 = y1 + dy1;
        float pred_x2 = x2 + dx2;
        float pred_y2 = y2 + dy2;

        pred_x1 = std::max(std::min(pred_x1, im_width - 1.0f), 0.0f);
        pred_y1 = std::max(std::min(pred_y1, im_height - 1.0f), 0.0f);
        pred_x2 = std::max(std::min(pred_x2, im_width - 1.0f), 0.0f);
        pred_y2 = std::max(std::min(pred_y2, im_height - 1.0f), 0.0f);

        (*out_pred_boxes)[index][0] = pred_x1;
        (*out_pred_boxes)[index][1] = pred_y1;
        (*out_pred_boxes)[index][2] = pred_x2;
        (*out_pred_boxes)[index][3] = pred_y2;
      }
    }
  }
}

// filter box by set confidence to zero
// * height or width < rpn_min_size
inline void FilterBox(mshadow::Tensor<cpu, 2>& dets,
                      const float min_size) {
  for (index_t i = 0; i < dets.size(0); i++) {
    float iw = dets[i][2] - dets[i][0] + 1.0f;
    float ih = dets[i][3] - dets[i][1] + 1.0f;
    if (iw < min_size || ih < min_size) {
      dets[i][0] -= min_size / 2;
      dets[i][1] -= min_size / 2;
      dets[i][2] += min_size / 2;
      dets[i][3] += min_size / 2;
      dets[i][4] = -1.0f;
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

//=====================
// NMS Utils
//=====================
namespace mxnet {
namespace op {
namespace utils {

struct ReverseArgsortCompl {
  const float *val_;
  explicit ReverseArgsortCompl(float *val)
    : val_(val) {}
  bool operator() (float i, float j) {
    return (val_[static_cast<index_t>(i)] >
            val_[static_cast<index_t>(j)]);
  }
};

// copy score and init order
inline void CopyScore(const mshadow::Tensor<cpu, 2>& dets,
                      mshadow::Tensor<cpu, 1>& score,
                      mshadow::Tensor<cpu, 1>& order) {
  for (index_t i = 0; i < dets.size(0); i++) {
    score[i] = dets[i][4];
    order[i] = i;
  }
}

// sort order array according to score
inline void ReverseArgsort(const mshadow::Tensor<cpu, 1>& score,
                           mshadow::Tensor<cpu, 1>& order) {
  ReverseArgsortCompl cmpl(score.dptr_);
  std::sort(order.dptr_, order.dptr_ + score.size(0), cmpl);
}

// reorder proposals according to order and keep the pre_nms_top_n proposals
// dets.size(0) == pre_nms_top_n
inline void ReorderProposals(const mshadow::Tensor<cpu, 2>& prev_dets,
                             const mshadow::Tensor<cpu, 1>& order,
                             const index_t pre_nms_top_n,
                             mshadow::Tensor<cpu, 2>& dets) {
  CHECK_EQ(dets.size(0), pre_nms_top_n);
  for (index_t i = 0; i < dets.size(0); i++) {
    const index_t index = order[i];
    for (index_t j = 0; j < dets.size(1); j++) {
      dets[i][j] = prev_dets[index][j];
    }
  }
}

// greedily keep the max detections (already sorted)
inline void NonMaximumSuppression(const mshadow::Tensor<cpu, 2>& dets,
                                  const float thresh,
                                  const index_t post_nms_top_n,
                                  mshadow::Tensor<cpu, 1>& area,
                                  mshadow::Tensor<cpu, 1>& suppressed,
                                  mshadow::Tensor<cpu, 1>& keep,
                                  index_t *out_size) {
  CHECK_EQ(dets.shape_[1], 5) << "dets: [x1, y1, x2, y2, score]";
  CHECK_GT(dets.shape_[0], 0);
  CHECK_EQ(dets.CheckContiguous(), true);
  CHECK_EQ(area.CheckContiguous(), true);
  CHECK_EQ(suppressed.CheckContiguous(), true);
  CHECK_EQ(keep.CheckContiguous(), true);
  // calculate area
  for (index_t i = 0; i < dets.size(0); ++i) {
    area[i] = (dets[i][2] - dets[i][0] + 1) *
              (dets[i][3] - dets[i][1] + 1);
  }

  // calculate nms
  *out_size = 0;
  for (index_t i = 0; i < dets.size(0) && (*out_size) < post_nms_top_n; ++i) {
    float ix1 = dets[i][0];
    float iy1 = dets[i][1];
    float ix2 = dets[i][2];
    float iy2 = dets[i][3];
    float iarea = area[i];

    if (suppressed[i] > 0.0f ) {
      continue;
    }

    keep[(*out_size)++] = i;
    for (index_t j = i + 1; j < dets.size(0); j ++) {
      if (suppressed[j] > 0.0f) {
        continue;
      }
      float xx1 = std::max(ix1, dets[j][0]);
      float yy1 = std::max(iy1, dets[j][1]);
      float xx2 = std::min(ix2, dets[j][2]);
      float yy2 = std::min(iy2, dets[j][3]);
      float w = std::max(0.0f, xx2 - xx1 + 1.0f);
      float h = std::max(0.0f, yy2 - yy1 + 1.0f);
      float inter = w * h;
      float ovr = inter / (iarea + area[j] - inter);
      if (ovr > thresh) {
        suppressed[j] = 1.0f;
      }
    }
  }
}

}  // namespace utils
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RCNN_UTILS_H_

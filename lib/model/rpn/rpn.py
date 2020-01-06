from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from lib.model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time


class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES # [8, 16, 32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS # [0.5, 1, 2]
        self.feat_stride = cfg.FEAT_STRIDE[0] # 16

        # define the convrelu layers processing input feature map
        # 卷积层输出 image 的 feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        # 先经过一个 1X1 的卷积层
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2  # 2(bg/fg) * 9 (anchors) = 18
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)  # 18 类 每个anchors可能是 positive negative

        # define anchor box offset prediction layer
        # 锚点偏移层
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4  # 4(coords) * 9 (anchors) = 36
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        """
        [batch_size, channel，height，width]
        [1, 2x9, H, W] => [1, d, H×2×9/d, W]

        :param x:
        :param d:
        :return:
        """
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)

        # get rpn classification score 通过softmax分类anchors获得positive和negative分类
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)  # [1, 18, H, W]

        # "Number of labels must match number of predictions; "
        # "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
        # "label count (number of labels) must be N*H*W, "
        # "with integer values in {0, 1, ..., C-1}."

        # 为什么在softmax前后都接一个reshape?
        # [1, 2x9, H, W] reshape layer 会变成 [1, 2, 9xH, W] 单独“腾空”出来一个维度以便softmax分类，之后再reshape回复原状
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)  # reshape 1 [1, 2, H*9, W]
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)  # [1, 2, H*9, W] softmax不学习参数
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)  # reshape 2  [1, 18, H, W]
        # 输出包含的信息 9个roi 每个roi是positive 还是 negative

        # get rpn offsets to the anchor boxes
        # 第二条主线 计算每个anchors的bbox regression偏移量 为什么还要计算negative的偏移量？
        # 该层输入图像为WxHx36 feature map的每个点都有9个anchors 每个anchor有4个回归的xywh变量
        #
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        # Region proposal
        # 吃的参数有 第一条主线的anchors分类器结果negative or positive
        #          第二条主线的anchors回归偏移量[dx(A), dy(A), dw(A), dh(A)] 整合之前的网络层，形成了RPN
        # 此处走Region_Proposal Layer 的 forward函数 输入rpn网络的一大堆roi
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            # data 有 labels bbox_targets bbox_inside_weight bbox_outside_weight

            # compute classification loss
            # 为什么分类误差拿的是第一个reshape(softmax之前)的数据做比较
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            # rpn region positive negative 分类误差
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            # 整个rpn网络的loss
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box

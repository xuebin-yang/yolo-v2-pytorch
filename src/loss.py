"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import math
import torch
import torch.nn as nn


class YoloLoss(nn.modules.loss._Loss):
    # The loss I borrow from LightNet repo.
    def __init__(self, num_classes, anchors, reduction=32, coord_scale=1.0, noobject_scale=1.0,
                 object_scale=5.0, class_scale=1.0, thresh=0.6):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)  # 每个格子预测 5 个bbox，5个类别
        self.anchor_step = len(anchors[0])  # 等于 2 指anchor base的宽高，等于4说明给的是左上角和右下角坐标
        self.anchors = torch.Tensor(anchors)
        self.reduction = reduction

        self.coord_scale = coord_scale  # 1.0
        self.noobject_scale = noobject_scale  # 1.0
        self.object_scale = object_scale  # 5.0
        self.class_scale = class_scale  # 1.0
        self.thresh = thresh            # 0.6

    def forward(self, output, target):  # output=[2, 125, 14, 14]  target=[aray[], array[]]

        batch_size = output.data.size(0)
        height = output.data.size(2)
        width = output.data.size(3)

        # Get x,y,w,h,conf,cls
        output = output.view(batch_size, self.num_anchors, -1, height * width)   # [2, 125, 14, 14]-->[2, 5, 25, 14*14] 每个格点（196个格点）都有5个预测框，然后每个预测框都有25个维度
        coord = torch.zeros_like(output[:, :, :4, :])  # 预测的 bbox 的坐标[2, 5, 4, 14*14]
        print('coord.shape', coord.shape)
        coord[:, :, :2, :] = output[:, :, :2, :].sigmoid()     # bbox坐标的 x，y, 用 sigmoid() 做了归一化, 将 14*14 的宽和高都当作 1
        coord[:, :, 2:4, :] = output[:, :, 2:4, :]  # 预测的 bbox 坐标的 w, h的偏移
        conf = output[:, :, 4, :].sigmoid()   # bbox 的置信度  [2, 5, 196] 里面的值代表每个格子预测的每个 anchor bbox 的置信度，做了sigmoid归一化
        cls = output[:, :, 5:, :].contiguous().view(batch_size * self.num_anchors, self.num_classes,   # 后面 20 个位置代表类别 [1960, 20]
                                                    height * width).transpose(1, 2).contiguous().view(-1,
                                                                                                      self.num_classes)
        print('cls.shape', cls.shape)   # [1960, 20]  又把所有框混到一起去了，前980个是第一张图片的预测情况

        # Create prediction boxes
        pred_boxes = torch.FloatTensor(batch_size * self.num_anchors * height * width, 4)   # [1960, 4] 即 bbox 的总数
        lin_x = torch.range(0, width - 1).repeat(height, 1).view(height * width)   # 列方向上的位置
        lin_y = torch.range(0, height - 1).repeat(width, 1).t().contiguous().view(height * width) # 行方向的位置
        anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)  # 预先置好的 anchor 的高
        anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)  # 预先治好的 anchor 的宽

        if torch.cuda.is_available():
            pred_boxes = pred_boxes.cuda()
            lin_x = lin_x.cuda()
            lin_y = lin_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)   # + lin_x 添加列信息，也就是加上在 x 方向上的偏移
        print('coord[:, :, 0].shape', coord[:, :, 0].shape)   # [2, 5, 196] 得到14*14的格点的每个框的相对中心点的 x 偏移
        print('pred_boxes.shape', coord.shape)   # [2, 5, 4, 196]
        pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)   #  得到14*14的格点的每个框的相对中心点的 y 偏移
        pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)  # 预测的bbox的宽高和faster RCNN一样，是相对于anchor宽高的一个放缩。exp(w)和exp(h)分别对应了宽高的放缩因子。
        pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
        pred_boxes = pred_boxes.cpu()   # pred_boxes 就是网络得到的特征（相对anchor_based的中心点和宽高偏差）和 anchor_based 的宽高之间得到的预测的框

        # Get target values
        coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(pred_boxes, target, height, width)
        # coord_mask 应该是要有obj的就会被置 1
        # conf_mask：应该存在 obj 的那个格子的第 best_n 个 anchor box 的置信度被置 5， 其他位置都为 1
        # cls_mask  应该存在obj的那个格子的第 best_n 个anchor置 1，代表着应该在这个格子的第 best_n 个 anchor 处有 obj
        # tcoord
        # tconf ： 预测的 bbox 和真实 bbox 的最大 iou
        # tcls

        coord_mask = coord_mask.expand_as(tcoord)
        print('coord_mask.shape', coord_mask.shape) # [56, 5, 4, 196]
        print('tcoord.shape', tcoord.shape) # [56, 5, 4, 196]
        tcls = tcls[cls_mask].view(-1).long()   # 挑出所有类别
        cls_mask = cls_mask.view(-1, 1).repeat(1, self.num_classes)

        if torch.cuda.is_available():
            tcoord = tcoord.cuda()
            tconf = tconf.cuda()
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            tcls = tcls.cuda()
            cls_mask = cls_mask.cuda()

        conf_mask = conf_mask.sqrt()
        cls = cls[cls_mask].view(-1, self.num_classes)

        # Compute losses
        mse = nn.MSELoss(size_average=False)
        ce = nn.CrossEntropyLoss(size_average=False)
        self.loss_coord = self.coord_scale * mse(coord * coord_mask, tcoord * coord_mask) / batch_size
        self.loss_conf = mse(conf * conf_mask, tconf * conf_mask) / batch_size
        self.loss_cls = self.class_scale * 2 * ce(cls, tcls) / batch_size
        self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls

        return self.loss_tot, self.loss_coord, self.loss_conf, self.loss_cls

    def build_targets(self, pred_boxes, ground_truth, height, width):
        batch_size = len(ground_truth)

        conf_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False) * self.noobject_scale  # [2, 5, 14*14]全 1， 每个patch有 5 个置信度
        coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False) # [2, 5, 1, 14*14] 全 0
        cls_mask = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False).byte() # [2, 5, 14*14] 全0
        tcoord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False) # [2, 5, 4, 14*14] 全0
        tconf = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False) # [2, 5, 14*14] 全0
        tcls = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False) # [2, 5, 14*14] 全0

        for b in range(batch_size):
            if len(ground_truth[b]) == 0:
                continue

            # Build up tensors
            cur_pred_boxes = pred_boxes[   # 当前一张图片的 bbox
                             b * (self.num_anchors * height * width):(b + 1) * (self.num_anchors * height * width)]
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)   # 将 anchor 中心移动到左上角00处
            gt = torch.zeros(len(ground_truth[b]), 4)  # [[0., 0., 0., 0.]]
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = (anno[0] + anno[2] / 2) / self.reduction   # 真实bbox的中心点，缩放后的中心点
                gt[i, 1] = (anno[1] + anno[3] / 2) / self.reduction   # 真实bbox的中心点，缩放后的中心点
                gt[i, 2] = anno[2] / self.reduction   # 真实标注框的在最后的特征图上的宽度
                gt[i, 3] = anno[3] / self.reduction   # 真实标注框的在最后的特征图上的高度  gt 表示真实的 bbox

            # Set confidence mask of matching detections to 0
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes)   # gt表示该张图片的真实的bbox，cur_pre_boxes表示该张图片预测出来的框。真实bbox和预测出来的框计算iou
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1  # [980]  # 预测框和任一个真实框的 iou 阈值高于 thresh 的框将被值为 true
            conf_mask[b][mask.view_as(conf_mask[b])] = 0  # 将 iou 阈值高于 thresh 的框的位置的置信度置 0， 也就是将某个框的第某个 anchor box 的位置置为 0（每个网格有 5 个置信度）
            print('mask.shape', mask.shape)   # [980]
            print('conf_mask[0].shape', conf_mask[0].shape)  # [5, 196]
            # Find best anchor for each ground truth
            gt_wh = gt.clone()   # 真实 bboxes
            gt_wh[:, :2] = 0    # 将真实 bboxes 的中心移动到左上角00处
            iou_gt_anchors = bbox_ious(gt_wh, anchors)   # 真实 bbox 与 anchor 的 iou， 中心坐标移动到 0， 0位置
            _, best_anchors = iou_gt_anchors.max(1)  # 找出先验框 anchor 里面和真实框 iou 最大的 anchor 是第几个，也就是确定框的形状

            # Set masks and target values for each ground truth
            for i, anno in enumerate(ground_truth[b]): # 逐个真实 bbox 考虑
                gi = min(width - 1, max(0, int(gt[i, 0])))   # bbox的中心在第 7 列
                gj = min(height - 1, max(0, int(gt[i, 1])))  # bbox的中心在第 8 行
                best_n = best_anchors[i]
                iou = iou_gt_pred[i][best_n * height * width + gj * width + gi]   # 按照真实bbox和anchor-base的iou最大的patch找到相应的patch和对应的先验框，能使得bbox的形状更合适
                coord_mask[b][best_n][0][gj * width + gi] = 1  #
                cls_mask[b][best_n][gj * width + gi] = 1  # 将应该存在obj的那个格子的第best_n 个anchor置 1，代表着应该在这个格子的第best_n个anchor处有obj
                conf_mask[b][best_n][gj * width + gi] = self.object_scale # 将应该存在 obj 的那个格子的第bast_n个 anchor box 的置信度置 5
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi  # 真实bbox中心距离网格边界的偏移量
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj  # 真实bbox的偏移量
                tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt[i, 2], 1.0) / self.anchors[best_n, 0])  # 做了偏移
                tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt[i, 3], 1.0) / self.anchors[best_n, 1])
                tconf[b][best_n][gj * width + gi] = iou  # 真实置信度为 iou
                tcls[b][best_n][gj * width + gi] = int(anno[4])  # 将存在 obj 的那个格子的第best_n个 anchor box 的类别置为真实类别

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls


def bbox_ious(boxes1, boxes2):
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions

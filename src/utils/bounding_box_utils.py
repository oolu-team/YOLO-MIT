import logging

import torch
from einops import rearrange
from torchvision.ops import batched_nms

from src.config.config import NMSConfig
from src.model.yolo import YOLO

logger = logging.getLogger("yolo")


def bbox_nms(
    cls_dist: torch.Tensor,
    bbox: torch.Tensor,
    nms_cfg: NMSConfig,
    confidence: torch.Tensor | None = None,
):
    cls_dist = cls_dist.sigmoid() * (1 if confidence is None else confidence)

    batch_idx, valid_grid, valid_cls = torch.where(cls_dist > nms_cfg.min_confidence)
    valid_con = cls_dist[batch_idx, valid_grid, valid_cls]
    valid_box = bbox[batch_idx, valid_grid]

    nms_idx = batched_nms(
        valid_box, valid_con, batch_idx + valid_cls * bbox.size(0), nms_cfg.min_iou
    )
    predicts_nms = []
    for idx in range(cls_dist.size(0)):
        instance_idx = nms_idx[idx == batch_idx[nms_idx]]

        predict_nms = torch.cat(
            [
                valid_cls[instance_idx][:, None],
                valid_box[instance_idx],
                valid_con[instance_idx][:, None],
            ],
            dim=-1,
        )

        predicts_nms.append(predict_nms[: nms_cfg.max_bbox])
    return predicts_nms


def generate_anchors(image_size: list[int], strides: list[int]):
    """
    Find the anchor maps for each w, h.

    Args:
        image_size List: the image size of augmented image size
        strides List[8, 16, 32, ...]: the stride size for each predicted layer

    Returns:
        all_anchors [HW x 2]:
        all_scalers [HW]: The index of the best targets for each anchors
    """
    W, H = image_size
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = W // stride * H // stride
        scaler.append(torch.full((anchor_num,), stride))
        shift = stride // 2
        h = torch.arange(0, H, stride) + shift
        w = torch.arange(0, W, stride) + shift
        if torch.__version__ >= "2.3.0":
            anchor_h, anchor_w = torch.meshgrid(h, w, indexing="ij")
        else:
            anchor_h, anchor_w = torch.meshgrid(h, w)
        anchor = torch.stack([anchor_w.flatten(), anchor_h.flatten()], dim=-1)
        anchors.append(anchor)
    all_anchors = torch.cat(anchors, dim=0)
    all_scalers = torch.cat(scaler, dim=0)
    return all_anchors, all_scalers


class Vec2Box:
    def __init__(self, strides: list[int], image_size, device):
        self.strides = strides
        self.device = device

        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = anchor_grid.to(device), scaler.to(device)

    @staticmethod
    def create_auto_anchor(model: YOLO, image_size):
        W, H = image_size
        # TODO: need accelerate dummy test
        dummy_input = torch.zeros(1, 3, H, W)
        dummy_output = model(dummy_input)
        strides = []
        for predict_head in dummy_output["Main"]:
            _, _, *anchor_num = predict_head[2].shape
            strides.append(W // anchor_num[1])
        return strides

    def update(self, image_size):
        """
        image_size: W, H
        """
        if self.image_size == image_size:
            return
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = (
            anchor_grid.to(self.device),
            scaler.to(self.device),
        )

    def __call__(self, predicts):
        preds_cls, preds_anc, preds_box = [], [], []
        for layer_output in predicts:
            pred_cls, pred_anc, pred_box = layer_output
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
            preds_anc.append(rearrange(pred_anc, "B A R h w -> B (h w) R A"))
            preds_box.append(rearrange(pred_box, "B X h w -> B (h w) X"))
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_anc = torch.concat(preds_anc, dim=1)
        preds_box = torch.concat(preds_box, dim=1)

        pred_LTRB = preds_box * self.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)
        return preds_cls, preds_anc, preds_box

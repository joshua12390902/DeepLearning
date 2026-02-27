import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Focal Loss
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: logits, any shape
        targets: {0,1} same shape as inputs
        """
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.sigmoid(inputs)
        p_t = targets * p + (1 - targets) * (1 - p)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =========================================================
# Box Loss (CIoU / GIoU / MSE) — 數值穩定版
# =========================================================
class BoxLoss(nn.Module):
    def __init__(self, loss_type: str = "ciou"):
        super().__init__()
        assert loss_type in ("ciou", "giou", "mse")
        self.type = loss_type

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, anchors):
        """
        pred_boxes: [B, H, W, A, 4] raw (tx, ty, tw, th)
        target_boxes: [B, H, W, A, 4] encoded (tx, ty, w, h) 其中 w,h 已是 [0,1]
        anchors: list/tuple[(w,h), ...] for this scale, each in [0,1]
        """
        B, G, _, A, _ = pred_boxes.shape
        device = pred_boxes.device
        dtype = pred_boxes.dtype

        # dtype-aware eps
        eps = torch.finfo(dtype).eps

        # anchors tensor
        anchors_t = torch.as_tensor(anchors, device=device, dtype=dtype).view(1, 1, 1, A, 2)

        # grid coords
        g_range = torch.arange(G, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(g_range, g_range, indexing="ij")  # [G,G]
        grid_xy = torch.stack([gx, gy], dim=-1).view(1, G, G, 1, 2)  # [1,G,G,1,2]

        if self.type in ("ciou", "giou"):
            # -------- 解碼預測框（比例座標）--------
            pred_xy = torch.sigmoid(pred_boxes[..., 0:2]) + grid_xy.expand_as(pred_boxes[..., 0:2])
            pred_xy = pred_xy / G 

            tw_th = torch.clamp(pred_boxes[..., 2:4], min=-10.0, max=10.0)
            pred_wh = torch.exp(tw_th) * anchors_t 
            pred_wh = torch.clamp(pred_wh, min=eps, max=1.0) # 確保 wh > 0

            pred_cxcywh = torch.cat([pred_xy, pred_wh], dim=-1)
            pred_x1y1 = torch.clamp(pred_cxcywh[..., 0:2] - pred_cxcywh[..., 2:4] / 2, 0.0, 1.0)
            pred_x2y2 = torch.clamp(pred_cxcywh[..., 0:2] + pred_cxcywh[..., 2:4] / 2, 0.0, 1.0)
            pred_xyxy = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

            # -------- 解碼目標框（比例座標）--------
            target_xy = (target_boxes[..., 0:2] + grid_xy.expand_as(target_boxes[..., 0:2])) / G
            target_wh = torch.clamp(target_boxes[..., 2:4], min=eps, max=1.0) # 確保 wh > 0

            tgt_cxcywh = torch.cat([target_xy, target_wh], dim=-1)
            tgt_x1y1 = torch.clamp(tgt_cxcywh[..., 0:2] - tgt_cxcywh[..., 2:4] / 2, 0.0, 1.0)
            tgt_x2y2 = torch.clamp(tgt_cxcywh[..., 0:2] + tgt_cxcywh[..., 2:4] / 2, 0.0, 1.0)
            tgt_xyxy = torch.cat([tgt_x1y1, tgt_x2y2], dim=-1)

            # -------- IoU（比例座標）--------
            ix1 = torch.max(pred_xyxy[..., 0], tgt_xyxy[..., 0])
            iy1 = torch.max(pred_xyxy[..., 1], tgt_xyxy[..., 1])
            ix2 = torch.min(pred_xyxy[..., 2], tgt_xyxy[..., 2])
            iy2 = torch.min(pred_xyxy[..., 3], tgt_xyxy[..., 3])

            inter_w = torch.clamp(ix2 - ix1, min=0.0)
            inter_h = torch.clamp(iy2 - iy1, min=0.0)
            inter_area = inter_w * inter_h

            pred_w = torch.clamp(pred_xyxy[..., 2] - pred_xyxy[..., 0], min=0.0)
            pred_h = torch.clamp(pred_xyxy[..., 3] - pred_xyxy[..., 1], min=0.0)
            tgt_w = torch.clamp(tgt_xyxy[..., 2] - tgt_xyxy[..., 0], min=0.0)
            tgt_h = torch.clamp(tgt_xyxy[..., 3] - tgt_xyxy[..., 1], min=0.0)

            pred_area = pred_w * pred_h
            tgt_area = tgt_w * tgt_h

            union = torch.clamp(pred_area + tgt_area - inter_area, min=eps)
            iou = torch.clamp(inter_area / union, 0.0, 1.0)

            # -------- 最小包圍框 C（比例座標）--------
            cx1 = torch.min(pred_xyxy[..., 0], tgt_xyxy[..., 0])
            cy1 = torch.min(pred_xyxy[..., 1], tgt_xyxy[..., 1])
            cx2 = torch.max(pred_xyxy[..., 2], tgt_xyxy[..., 2])
            cy2 = torch.max(pred_xyxy[..., 3], tgt_xyxy[..., 3])

            c_w = torch.clamp(cx2 - cx1, min=eps)
            c_h = torch.clamp(cy2 - cy1, min=eps)

        # -------- CIoU --------
        if self.type == "ciou":
            c2 = c_w.pow(2) + c_h.pow(2) + eps
            rho2 = (pred_cxcywh[..., 0] - tgt_cxcywh[..., 0]).pow(2) + \
                   (pred_cxcywh[..., 1] - tgt_cxcywh[..., 1]).pow(2)
            rho_term = rho2 / c2

            # --- [*** 最終 NaN 修正 (V3) ***] ---
            # (這是 03:06 AM 的版本，確保 atan() 穩定)
            v = (4.0 / (torch.pi ** 2)) * torch.pow(
                torch.atan(target_wh[..., 0] / target_wh[..., 1]) -
                torch.atan(pred_wh[..., 0]   / pred_wh[..., 1]), 2
            )
            # --- [*** 修正完畢 ***] ---

            with torch.no_grad():
                alpha = v / (1.0 - iou + v + eps)

            ciou = 1.0 - iou + rho_term + alpha * v
            return ciou  # [B,H,W,A]

        # -------- GIoU --------
        if self.type == "giou":
            c_area = c_w * c_h 
            giou = iou - (c_area - union) / torch.clamp(c_area, min=eps)
            return 1.0 - giou  # [B,H,W,A]

        # -------- MSE（raw 空間）--------
        assert self.type == "mse"
        pred_tx = torch.sigmoid(pred_boxes[..., 0])
        pred_ty = torch.sigmoid(pred_boxes[..., 1])
        pred_tw = pred_boxes[..., 2]
        pred_th = pred_boxes[..., 3]

        tgt_tx = target_boxes[..., 0]
        tgt_ty = target_boxes[..., 1]
        tgt_tw = torch.log(target_boxes[..., 2] / (anchors_t[..., 0] + eps))
        tgt_th = torch.log(target_boxes[..., 3] / (anchors_t[..., 1] + eps))

        loss_tx = F.mse_loss(pred_tx, tgt_tx, reduction="none")
        loss_ty = F.mse_loss(pred_ty, tgt_ty, reduction="none")
        loss_tw = F.mse_loss(pred_tw, tgt_tw, reduction="none")
        loss_th = F.mse_loss(pred_th, tgt_th, reduction="none")
        return loss_tx + loss_ty + loss_tw + loss_th


# =========================================================
# YOLOv3 Loss（使用上面的 BoxLoss + FocalLoss）
# =========================================================
class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        box_loss_type: str = "ciou",
        lambda_coord: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.3,
        lambda_class: float = 0.75,
        anchors=None,
        # --- [*** 您的修改點 ***] ---
        use_focal_for_obj_cls: bool = False, # <-- 預設改回 False (使用 BCE)
        # --- [*** 修改完畢 ***] ---
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="none")
        self.use_focal = use_focal_for_obj_cls

        self.box_loss_fn = BoxLoss(loss_type=box_loss_type)
        self.anchors = anchors 

    def forward(self, predictions, targets):
        """
        predictions: list of 3 scales, each [B, G, G, 3*(5+C)]
        targets:     list of 3 scales, each [B, G, G, 3, 5+C]
        """
        device = predictions[0].device

        total_box = torch.tensor(0.0, device=device)
        total_obj_pos = torch.tensor(0.0, device=device)
        total_obj_neg = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)

        npos = 0
        nneg = 0

        for pred, gt, anchors in zip(predictions, targets, self.anchors):
            B, G, _, A, Cplus5 = gt.shape  # gt 是 [B,G,G,3,5+C]
            if pred.dim() == 4 and pred.shape[-1] == A * Cplus5:
                pred = pred.view(B, G, G, A, Cplus5)

            pred_boxes = pred[..., :4]      # (tx, ty, tw, th)
            pred_obj = pred[..., 4]         # logits
            pred_cls = pred[..., 5:]        # logits
            tgt_boxes = gt[..., :4]         # (tx, ty, w, h) w,h in [0,1]
            tgt_obj = gt[..., 4]            # {0,1}
            tgt_cls = gt[..., 5:]           # one-hot

            obj_mask = (tgt_obj == 1.0)
            noobj_mask = (tgt_obj == 0.0)

            # Box loss（只在正樣本上）
            box_loss = self.box_loss_fn(pred_boxes, tgt_boxes, anchors)
            box_loss = (box_loss * obj_mask).sum()

            # Objectness loss（正負樣本分開加權）
            if self.use_focal:
                obj_loss_all = self.focal(pred_obj, tgt_obj)
            else:
                obj_loss_all = self.bce(pred_obj, tgt_obj) # <-- 將會執行這裡

            obj_loss_pos = (obj_loss_all * obj_mask).sum()
            obj_loss_neg = (obj_loss_all * noobj_mask).sum()

            # Class loss（只在正樣本上）
            if pred_cls.numel() > 0:
                if self.use_focal:
                    cls_loss_all = self.focal(pred_cls, tgt_cls)
                else:
                    cls_loss_all = self.bce(pred_cls, tgt_cls) # <-- 將會執行這裡
                cls_loss = (cls_loss_all * obj_mask.unsqueeze(-1)).sum()
            else:
                cls_loss = torch.tensor(0.0, device=device)

            total_box += box_loss
            total_obj_pos += obj_loss_pos
            total_obj_neg += obj_loss_neg
            total_cls += cls_loss

            npos += int(obj_mask.sum().item())
            nneg += int(noobj_mask.sum().item())

        # 防止除零
        pos_denom = max(npos, 1)
        neg_denom = max(nneg, 1)

        box_avg = total_box / pos_denom
        obj_avg = total_obj_pos / pos_denom
        cls_avg = total_cls / pos_denom
        noobj_avg = total_obj_neg / neg_denom

        total = (
            self.lambda_coord * box_avg
            + self.lambda_obj * obj_avg
            + self.lambda_noobj * noobj_avg
            + self.lambda_class * cls_avg
        )

        return {
            "total": total,
            "box": box_avg,
            "obj": obj_avg,
            "noobj": noobj_avg,
            "cls": cls_avg,
        }
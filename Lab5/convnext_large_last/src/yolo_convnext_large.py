import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.config_new import GRID_SIZES,ANCHORS,YOLO_IMG_DIM



# =======================================================================
# Backbone
# =======================================================================
class Backbone(nn.Module):
    """
    Timm backbone (features_only=True)
    forward 回傳最後三層特徵：f5(最小/最深), f4, f3(最大/淺層)
    並暴露 c3,c4,c5 供 Head/Neck 使用
    """
    # ✅ 改成 convnextv2_large.fcmae_ft_in1k
    def __init__(self, model_name="convnextv2_large.fcmae_ft_in1k", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3)   # 一樣拿 C3, C4, C5
        )
        chs = self.backbone.feature_info.channels()

        # chs[0] = c3, chs[1] = c4, chs[2] = c5
        self.c3, self.c4, self.c5 = chs[0], chs[1], chs[2]

    def forward(self, x):
        feats = self.backbone(x)
        # feats[0] = c3 (大特徵圖)
        # feats[1] = c4 (中特徵圖)
        # feats[2] = c5 (小特徵圖)
        return feats[2], feats[1], feats[0]   # f5=c5, f4=c4, f3=c3


# =======================================================================
# 基礎積木
# =======================================================================
class ConvBlock(nn.Module):
    """
    標準 ConvBlock (Conv+BN+SiLU)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    CSPDarknet 用的基礎瓶頸層 (c=in)
    """
    def __init__(self, c_in, c_out, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        c_mid = int(c_out * expansion)
        self.conv1 = ConvBlock(c_in, c_mid, 1, 1)
        self.conv2 = ConvBlock(c_mid, c_out, 3, 1, 1, groups=groups)
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class CSP(nn.Module):
    """
    CSPNet 核心 (YOLOv5 C3)
    """
    def __init__(self, c_in, c_out, n_bottlenecks=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        c_mid = int(c_out * expansion)
        self.conv1 = ConvBlock(c_in, c_mid, 1, 1)
        self.conv2 = ConvBlock(c_in, c_mid, 1, 1)
        self.conv3 = ConvBlock(2 * c_mid, c_out, 1, 1)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(c_mid, c_mid, shortcut, groups, expansion=1.0) for _ in range(n_bottlenecks)]
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y1 = self.bottlenecks(y1)
        y2 = self.conv2(x)
        return self.conv3(torch.cat([y1, y2], dim=1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling Fast (YOLOv5)
    """
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_mid = c_in // 2
        self.conv1 = ConvBlock(c_in, c_mid, 1, 1)
        self.conv2 = ConvBlock(c_mid * 4, c_out, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], 1))


# =======================================================================
# Attention Modules: SE, ECA, CBAM, ClassAttention
# =======================================================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA)
    簡化版：global avg + 1D conv
    """
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)          # (B, C, 1, 1)
        y = y.view(b, 1, c)           # (B, 1, C)
        y = self.conv(y)              # (B, 1, C)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_sa = self.conv(x_cat)
        return self.sigmoid(x_sa)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ClassAttentionBlock(nn.Module):
    """
    簡化版 Class-Aware Attention：
    專門用在分類分支的 feature 上類似 SE 但參數獨立。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# =======================================================================
# ASFF: Adaptive Spatial Feature Fusion
# =======================================================================
class ASFFLevel(nn.Module):
    """
    對三個尺度 (small, mid, large) 做自適應空間融合，輸出指定 level 的 feature。
    level: 0 -> large, 1 -> mid, 2 -> small
    ch_s, ch_m, ch_l: 三個尺度輸入 channels
    """
    def __init__(self, level, ch_s, ch_m, ch_l):
        super().__init__()
        self.level = level

        # 對齊到同一個中間 channels
        if level == 0:       # large
            inter_dim = ch_l
        elif level == 1:     # mid
            inter_dim = ch_m
        else:                # small
            inter_dim = ch_s

        self.inter_dim = inter_dim

        self.conv_s = ConvBlock(ch_s, inter_dim, 1, 1)
        self.conv_m = ConvBlock(ch_m, inter_dim, 1, 1)
        self.conv_l = ConvBlock(ch_l, inter_dim, 1, 1)

        # 權重預測
        self.weight_conv = nn.Conv2d(inter_dim * 3, 3, kernel_size=1, stride=1, padding=0)
        # 融合後再調整一下
        self.output_conv = ConvBlock(inter_dim, inter_dim, 1, 1)

    def forward(self, f_s, f_m, f_l):
        # f_s: small, f_m: mid, f_l: large
        if self.level == 0:  # large 的輸出，以 f_l 尺寸為基準
            size = f_l.shape[-2:]
            s = F.interpolate(self.conv_s(f_s), size=size, mode='nearest')
            m = F.interpolate(self.conv_m(f_m), size=size, mode='nearest')
            l = self.conv_l(f_l)
        elif self.level == 1:  # mid
            size = f_m.shape[-2:]
            s = F.interpolate(self.conv_s(f_s), size=size, mode='nearest')
            m = self.conv_m(f_m)
            l = F.interpolate(self.conv_l(f_l), size=size, mode='nearest')
        else:  # small
            size = f_s.shape[-2:]
            s = self.conv_s(f_s)
            m = F.interpolate(self.conv_m(f_m), size=size, mode='nearest')
            l = F.interpolate(self.conv_l(f_l), size=size, mode='nearest')

        cat = torch.cat([s, m, l], dim=1)
        weight = self.weight_conv(cat)
        weight = F.softmax(weight, dim=1)

        s_w = weight[:, 0:1, :, :]
        m_w = weight[:, 1:2, :, :]
        l_w = weight[:, 2:3, :, :]

        fused = s * s_w + m * m_w + l * l_w
        out = self.output_conv(fused)
        return out


class ASFF(nn.Module):
    """
    三個 level 的 ASFF 一起輸出 (small, mid, large) 三個尺度的融合特徵。
    """
    def __init__(self, ch_s, ch_m, ch_l):
        super().__init__()
        self.level0 = ASFFLevel(level=0, ch_s=ch_s, ch_m=ch_m, ch_l=ch_l)  # large
        self.level1 = ASFFLevel(level=1, ch_s=ch_s, ch_m=ch_m, ch_l=ch_l)  # mid
        self.level2 = ASFFLevel(level=2, ch_s=ch_s, ch_m=ch_m, ch_l=ch_l)  # small

    def forward(self, features):
        # features: (f_s, f_m, f_l) = (small, mid, large)
        f_s, f_m, f_l = features
        out_l = self.level0(f_s, f_m, f_l)  # large
        out_m = self.level1(f_s, f_m, f_l)  # mid
        out_s = self.level2(f_s, f_m, f_l)  # small
        # 維持「small, mid, large」順序
        return out_s, out_m, out_l


# =======================================================================
# SOTA Neck (PANet) & Head (Decoupled)
# =======================================================================
class CSPNeck(nn.Module):
    """
    SOTA Neck: PANet (Path Aggregation Network)
    使用 CSP block 進行特徵融合 (動態適應 backbone channels)
    """
    def __init__(self, c3_ch, c4_ch, c5_ch, n_csp_blocks=1):
        super().__init__()
        
        # (Neck 內部的 channels 固定為 128, 256, 512)
        neck_c3_out = 128
        neck_c4_out = 256
        neck_c5_out = 512

        # Top-Down (FPN part)
        # (c5_ch -> neck_c5_out) e.g. 768/1024 -> 512
        self.up_sppf = SPPF(c5_ch, neck_c5_out)
        # (neck_c5_out -> neck_c4_out) e.g. 512 -> 256
        self.up_conv1 = ConvBlock(neck_c5_out, neck_c4_out, 1, 1)
        # (c4_ch + neck_c4_out -> neck_c4_out)
        self.up_csp1 = CSP(c4_ch + neck_c4_out, neck_c4_out, n_csp_blocks, shortcut=False)
        # (neck_c4_out -> neck_c3_out) e.g. 256 -> 128
        self.up_conv2 = ConvBlock(neck_c4_out, neck_c3_out, 1, 1)
        # (c3_ch + neck_c3_out -> neck_c3_out)
        self.up_csp2 = CSP(c3_ch + neck_c3_out, neck_c3_out, n_csp_blocks, shortcut=False)

        # Bottom-Up (PAN part)
        self.down_conv1 = ConvBlock(neck_c3_out, neck_c3_out, 3, 2, 1) # P3 -> P4
        self.down_csp3 = CSP(neck_c4_out + neck_c3_out, neck_c4_out, n_csp_blocks, shortcut=False)
        self.down_conv2 = ConvBlock(neck_c4_out, neck_c4_out, 3, 2, 1) # P4 -> P5
        self.down_csp4 = CSP(neck_c5_out + neck_c4_out, neck_c5_out, n_csp_blocks, shortcut=False)

        # 把 channel 大小記起來給 ASFF / Head 用
        self.neck_c3_out = neck_c3_out  # large
        self.neck_c4_out = neck_c4_out  # mid
        self.neck_c5_out = neck_c5_out  # small

    def forward(self, features):
        f5, f4, f3 = features # f5=small(c5), f4=mid(c4), f3=large(c3)
        
        # FPN (Top-Down)
        p5 = self.up_sppf(f5) 
        p4_in = self.up_conv1(p5) 
        p4_in = F.interpolate(p4_in, size=f4.shape[2:], mode='nearest')
        p4 = self.up_csp1(torch.cat([f4, p4_in], dim=1))
        
        p3_in = self.up_conv2(p4) 
        p3_in = F.interpolate(p3_in, size=f3.shape[2:], mode='nearest')
        p3 = self.up_csp2(torch.cat([f3, p3_in], dim=1))
        
        # PAN (Bottom-Up)
        n4_in = self.down_conv1(p3)
        n4 = self.down_csp3(torch.cat([n4_in, p4], dim=1))
        
        n5_in = self.down_conv2(n4)
        n5 = self.down_csp4(torch.cat([n5_in, p5], dim=1))
        
        # 回傳 channels: (neck_c5_out, neck_c4_out, neck_c3_out)
        # small(512), mid(256), large(128)
        return n5, n4, p3


class DecoupleHeadUnit(nn.Module):
    """
    SOTA Head: YOLOX Decoupled Head + Attention
    (reg, obj, cls)
    """
    def __init__(self, in_ch: int, na: int, nc: int, act=nn.SiLU(inplace=True)):
        super().__init__()
        self.na, self.nc = na, nc
        self.act = act

        # ------- 回歸 / 物件分支 -------
        self.reg_conv = nn.Sequential(
            ConvBlock(in_ch, in_ch, 3, 1, 1),
            ConvBlock(in_ch, in_ch, 3, 1, 1),
        )
        # 注意力：SE + ECA + CBAM
        self.reg_se  = SEBlock(in_ch)
        self.reg_eca = ECALayer(in_ch)
        self.reg_cbam = CBAMBlock(in_ch)

        self.reg_out = nn.Conv2d(in_ch, na * 4, 1, 1, 0)
        self.obj_out = nn.Conv2d(in_ch, na * 1, 1, 1, 0)

        # ------- 分類分支 -------
        self.cls_conv = nn.Sequential(
            ConvBlock(in_ch, in_ch, 3, 1, 1),
            ConvBlock(in_ch, in_ch, 3, 1, 1),
        )
        self.cls_se   = SEBlock(in_ch)
        self.cls_eca  = ECALayer(in_ch)
        self.cls_cbam = CBAMBlock(in_ch)
        self.cls_class_attn = ClassAttentionBlock(in_ch)

        self.cls_out = nn.Conv2d(in_ch, na * nc, 1, 1, 0)

        self._init_bias()

    def _init_bias(self):
        with torch.no_grad():
            b = self.obj_out.bias.view(self.na, 1)
            b.fill_(-4.5)
            self.obj_out.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            b = self.cls_out.bias.view(self.na, self.nc)
            b.fill_(-4.5)
            self.cls_out.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        # ----- 回歸 / obj -----
        r = self.reg_conv(x)
        r = self.reg_se(r)
        r = self.reg_eca(r)
        r = self.reg_cbam(r)
        reg = self.reg_out(r)
        obj = self.obj_out(r)

        # ----- 分類 -----
        c = self.cls_conv(x)
        c = self.cls_se(c)
        c = self.cls_eca(c)
        c = self.cls_cbam(c)
        c = self.cls_class_attn(c)
        cls = self.cls_out(c)

        # (B, na*(4+1+nc), H, W)
        return torch.cat([reg, obj, cls], dim=1)


class ODHead(nn.Module):
    """
    組合 3 個 DecoupleHeadUnit
    """
    def __init__(self, ch_s, ch_m, ch_l, num_anchors, num_classes):
        super().__init__()
        self.dec_s = DecoupleHeadUnit(ch_s, num_anchors, num_classes)
        self.dec_m = DecoupleHeadUnit(ch_m, num_anchors, num_classes)
        self.dec_l = DecoupleHeadUnit(ch_l, num_anchors, num_classes)
    
    def forward(self, features):
        f_s, f_m, f_l = features # small, mid, large
        
        p_s = self.dec_s(f_s)
        p_m = self.dec_m(f_m)
        p_l = self.dec_l(f_l)

        # (B, C, H, W) -> (B, H, W, C)
        p_s = p_s.permute(0, 2, 3, 1).contiguous()
        p_m = p_m.permute(0, 2, 3, 1).contiguous()
        p_l = p_l.permute(0, 2, 3, 1).contiguous()
        
        return p_s, p_m, p_l # small, mid, large


# ============================================================================
# NMS for inference 
# ============================================================================
def non_max_suppression(prediction, conf_thres=0.3, nms_thres=0.4):
    """
    prediction: (B, N_all_anchors, 5 + num_classes)
    """
    from torchvision.ops import batched_nms
    output = [None for _ in range(len(prediction))]
    
    for image_i, image_pred in enumerate(prediction):
        # 1. 過濾
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
        obj_conf = image_pred[:, 4]
        combined_conf = obj_conf * class_conf.squeeze()
        
        obj_mask = (obj_conf >= conf_thres * 0.1) # 預篩選
        image_pred = image_pred[obj_mask]
        class_conf = class_conf[obj_mask]
        class_pred = class_pred[obj_mask]
        obj_conf = obj_conf[obj_mask]
        combined_conf = combined_conf[obj_mask]

        if image_pred.numel() == 0:
            continue
            
        conf_mask = (combined_conf >= conf_thres) # 細篩選
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        obj_conf = obj_conf[conf_mask]
        
        if image_pred.numel() == 0:
            continue

        # 2. 轉座標 (cx,cy,w,h) -> (x1,y1,x2,y2)
        boxes_xyxy = image_pred[:, :4].clone()
        boxes_xyxy[:, 0] = image_pred[:, 0] - image_pred[:, 2] / 2
        boxes_xyxy[:, 1] = image_pred[:, 1] - image_pred[:, 3] / 2
        boxes_xyxy[:, 2] = image_pred[:, 0] + image_pred[:, 2] / 2
        boxes_xyxy[:, 3] = image_pred[:, 1] + image_pred[:, 3] / 2
        
        scores = obj_conf * class_conf.squeeze(-1)
        labels = class_pred.squeeze(-1)
        
        # 3. NMS
        keep_indices = batched_nms(boxes_xyxy, scores, labels, nms_thres)
        
        # 4. 格式化輸出
        output[image_i] = torch.cat([
            image_pred[keep_indices, :5],
            class_conf[keep_indices].float(),
            class_pred[keep_indices].float()
        ], dim=1)
        
    return output


# ============================================================================
# Object Detection Model (SOTA 組合)
# ============================================================================
class ODModel(nn.Module):
    """
    SOTA Model: (Timm) Backbone + PANet + ASFF + Decoupled Head + Attention
    """
    def __init__(self, num_classes=20, num_anchors=3, pretrained=True, 
                 nms_thres=0.4, conf_thres=0.5,
                 model_name="convnextv2_large.fcmae_ft_in1k"):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.backbone = Backbone(pretrained=pretrained, model_name=model_name)
        
        self.neck = CSPNeck(
            self.backbone.c3, # e.g. 256
            self.backbone.c4, # e.g. 512
            self.backbone.c5  # e.g. 1024
        )

        # ASFF：融合 neck 輸出三個尺度
        self.asff = ASFF(
            ch_s=self.neck.neck_c5_out,  # 512 (small)
            ch_m=self.neck.neck_c4_out,  # 256 (mid)
            ch_l=self.neck.neck_c3_out   # 128 (large)
        )
        
        # Head：使用 ASFF 融合後的三個尺度特徵
        self.head = ODHead(
            ch_s=self.neck.neck_c5_out, # n5 (small)
            ch_m=self.neck.neck_c4_out, # n4 (mid)
            ch_l=self.neck.neck_c3_out, # p3 (large)
            num_anchors=num_anchors, 
            num_classes=num_classes
        )
        
        self.anchors = ANCHORS
        self.strides = torch.tensor([8, 16, 32]) # (動態 Grid 計算用)
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres
        
        self._initialize_biases()

    def _initialize_biases(self):
        with torch.no_grad():
            for m in self.head.modules():
                if isinstance(m, DecoupleHeadUnit):
                    m._init_bias() 

    def forward(self, x):
        features = self.backbone(x)
        features_neck = self.neck(features)
        # ASFF 融合
        features_fused = self.asff(features_neck)
        predictions = self.head(features_fused)
        return predictions # (small, mid, large)

    def _transform_predictions(self, pred, anchors):
        # (動態 grid_size)
        batch_size = pred.size(0)
        grid_size_h, grid_size_w = pred.shape[1:3] # (G_H, G_W)
        
        pred = pred.view(batch_size, grid_size_h, grid_size_w, self.num_anchors, 5 + self.num_classes)
        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]
        obj_conf = torch.sigmoid(pred[..., 4])
        cls_conf = torch.sigmoid(pred[..., 5:])

        # (動態 grid_x, grid_y)
        grid_x = torch.arange(grid_size_w, dtype=torch.float, device=pred.device).view(1, 1, grid_size_w, 1).repeat(1, grid_size_h, 1, 1)
        grid_y = torch.arange(grid_size_h, dtype=torch.float, device=pred.device).view(1, grid_size_h, 1, 1).repeat(1, 1, grid_size_w, 1)

        anchor_w = torch.tensor([a[0] for a in anchors], dtype=torch.float, device=pred.device).view(1, 1, 1, self.num_anchors)
        anchor_h = torch.tensor([a[1] for a in anchors], dtype=torch.float, device=pred.device).view(1, 1, 1, self.num_anchors)

        w_clamped = torch.clamp(w, -10, 10)
        h_clamped = torch.clamp(h, -10, 10)

        pred_boxes = torch.zeros_like(pred[..., :4])
        # (正規化到 0-1)
        pred_boxes[..., 0] = (x + grid_x) / grid_size_w
        pred_boxes[..., 1] = (y + grid_y) / grid_size_h
        pred_boxes[..., 2] = torch.exp(w_clamped) * anchor_w
        pred_boxes[..., 3] = torch.exp(h_clamped) * anchor_h

        output = torch.cat([pred_boxes, obj_conf.unsqueeze(-1), cls_conf], dim=-1)
        return output

    def get_pre_nms_boxes(self, x):
        """
        SOTA 版: 取得 pre-NMS 框 (TTA 必須)
        """
        batch_size = x.size(0)
        features = self.backbone(x)
        features_neck = self.neck(features)
        features_fused = self.asff(features_neck)
        pred_s, pred_m, pred_l = self.head(features_fused) 
        
        p_s = self._transform_predictions(pred_s, self.anchors[0])
        p_m = self._transform_predictions(pred_m, self.anchors[1])
        p_l = self._transform_predictions(pred_l, self.anchors[2])
        
        all_predictions = []
        for i in range(batch_size):
            pred_i = torch.cat([
                p_l[i].view(-1, 5 + self.num_classes), # large
                p_m[i].view(-1, 5 + self.num_classes), # mid
                p_s[i].view(-1, 5 + self.num_classes)  # small
            ], dim=0)
            all_predictions.append(pred_i)
        
        return torch.stack(all_predictions, dim=0) 

    def inference(self, x, conf_thres=None, nms_thres=None):
        if conf_thres is None: conf_thres = self.conf_thres
        if nms_thres is None: nms_thres = self.nms_thres
        
        self.eval()
        with torch.no_grad():
            all_predictions = self.get_pre_nms_boxes(x)
            output = non_max_suppression(all_predictions, conf_thres, nms_thres)
            
        return output


def getODmodel(pretrained=True):
    # 預設使用 convnext_base.fb_in1k
    model = ODModel(num_classes=20, num_anchors=3, pretrained=pretrained)
    return model

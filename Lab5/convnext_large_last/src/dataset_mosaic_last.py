import os
import random
import multiprocessing as mp
from collections import defaultdict # 
import cv2
import numpy as np
import torch
import torch.utils.data as DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config_new import (
    VOC_IMG_MEAN,
    VOC_IMG_STD,
    YOLO_IMG_DIM,
    ANCHORS,
    GRID_SIZES,
)


# ============================================================
# 
# ============================================================

test_data_pipelines = A.Compose([
    A.Resize(height=YOLO_IMG_DIM, width=YOLO_IMG_DIM),
    A.Normalize(mean=VOC_IMG_MEAN, std=VOC_IMG_STD),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls_labels'], min_visibility=0.1))


train_data_pipelines = A.Compose([
    # --- 
    A.HorizontalFlip(p=0.5),
    A.Affine(
        translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
        scale=(0.9, 1.1), # (
        rotate=(-10, 10), # (SOTA  0.0)
        p=0.5,
        border_mode=cv2.BORDER_CONSTANT,
        value=114 # (YOLOv5  114)
    ),
    
    # --- 
    # ( 03:22 baseline )
    A.RandomBrightnessContrast(p=0.20),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.30),
    A.RGBShift(p=0.10),
    
    # --- 
    A.PixelDropout(dropout_prob=0.03, per_channel=False, drop_value=0, p=0.20),

    # --- 
    A.Normalize(mean=VOC_IMG_MEAN, std=VOC_IMG_STD),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls_labels'], min_visibility=0.1))


# ============================================================
# SOTA Dataset (Mosaic + Mixup)
# ============================================================
class VocDetectorDataset(DataLoader.Dataset):
    image_size = YOLO_IMG_DIM

    def __init__(
        self,
        root_img_dir,
        dataset_file,
        train,
        contain_labels=True,
        num_classes=20,
        grid_sizes=GRID_SIZES,
        transform=None, # (transform  'train_data_pipelines')
        return_image_id=False,
        encode_target=True,
    ):
        print(f"Initializing dataset (train={train})")
        self.root = root_img_dir
        self.contain_labels = contain_labels
        self.train = train
        self.fnames, self.boxes, self.labels = [], [], []
        self.grid_sizes = grid_sizes
        self.num_classes = num_classes
        self.return_image_id = return_image_id
        self.encode_target = encode_target
        
        # --- [SOTA] Mosaic & Mixup 
        self.enable_mosaic = self.train
        self.mosaic_prob = 1.0   # ( train_v3.py )
        self.mixup_prob = 0.15   # (SOTA  0.15)
        self.mixup_beta = 32.0   # (SOTA  32.0,  NaN)
        # ---
        
        self.augment = train_data_pipelines if (train and transform is None) else test_data_pipelines
        self.test_pipe = test_data_pipelines
        
        # (
        self.random_crop = A.RandomSizedBBoxSafeCrop(
            width=self.image_size, height=self.image_size, erosion_rate=0, p=1.0
        )
        self.base_resize = A.Resize(height=self.image_size, width=self.image_size)
        
        # ( 12:51 AM  Bug)
        self.crop_transform = A.Compose(
            [self.random_crop], 
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls_labels'], min_visibility=0.1)
        )
        self.resize_transform = A.Compose(
            [self.base_resize],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls_labels'], min_visibility=0.1)
        )

        with open(dataset_file) as f:
            lines = f.readlines()
        for line in lines:
            if self.contain_labels is False:
                continue
            split_line = line.strip().split()
            self.fnames.append(split_line[0])
            num_boxes = (len(split_line) - 1) // 5
            box, label = [], []
            for i in range(num_boxes):
                x1 = float(split_line[1 + 5 * i])
                y1 = float(split_line[2 + 5 * i])
                x2 = float(split_line[3 + 5 * i])
                y2 = float(split_line[4 + 5 * i])
                c  = split_line[5 + 5 * i]
                box.append([x1, y1, x2, y2])
                label.append(int(c))
            self.boxes.append(box)
            self.labels.append(label)
        self.num_samples = len(self.boxes)
        
        # --- [修改點：Class-aware Sampling] ---
        # 
        # (
        self.class_img_map = defaultdict(list)
        if self.train:
            print("Building class-to-image map for Class-Aware Sampling...")
            for i in range(self.num_samples):
                img_labels = set(self.labels[i])
                for cls_idx in img_labels:
                    self.class_img_map[cls_idx].append(i)
            print("Class-to-image map built.")
        # --- [修改結束] ---

    
    def set_input_dim(self, new_dim: int):
        """
        (!!!  SOTA Multi-Scale 
        
        """
        # 1.  image_size
        self.image_size = new_dim
        
        # 2. (
        
        # (
        self.random_crop = A.RandomSizedBBoxSafeCrop(
            width=self.image_size, height=self.image_size, erosion_rate=0, p=1.0
        )
        self.base_resize = A.Resize(height=self.image_size, width=self.image_size)
        
        # ( 12:51 AM  Bug)
        self.crop_transform = A.Compose(
            [self.random_crop], 
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls_labels'], min_visibility=0.1)
        )
        self.resize_transform = A.Compose(
            [self.base_resize],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls_labels'], min_visibility=0.1)
        )
    # ========================================

    def _compute_iou_wh(self, w1, h1, w2, h2):
        inter = min(w1, w2) * min(h1, h2)
        union = w1 * h1 + w2 * h2 - inter
        return (inter / (union + 1e-16))

    def from_cxcy_to_gridxy(self, cx, cy, grid_size, image_size):
        stride_x = image_size[0] / grid_size
        stride_y = image_size[1] / grid_size
        grid_x, tx = divmod(cx, stride_x)
        grid_y, ty = divmod(cy, stride_y)
        grid_x = int(grid_x); grid_y = int(grid_y)
        tx, ty = tx / stride_x, ty / stride_y
        return grid_x, grid_y, tx, ty

    def encoder(self, image, boxes:list, labels:list):
        # ( 03:48  encoder， IoU > 0.3)
        # (
        num_scales = len(self.grid_sizes)
        H, W = image.shape[1], image.shape[2]
        target_boxes = [torch.zeros(gs, gs, 3, 4) for gs in self.grid_sizes]
        target_cls   = [torch.zeros(gs, gs, 3, self.num_classes) for gs in self.grid_sizes]
        target_obj   = [torch.zeros(gs, gs, 3) for gs in self.grid_sizes]
        anchors = torch.tensor(ANCHORS) # (3, 3, 2)
        
        smooth = 0.0 # (Label smoothing)

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) * 0.5 / W
            cy = (y1 + y2) * 0.5 / H
            w  = (x2 - x1) / W
            h  = (y2 - y1) / H
            
            if w <= 0 or h <= 0: continue
            
            # (SOTA 
            # ( "best_iou" 
            
            for s in range(num_scales):
                gs = self.grid_sizes[s]
                gx, gy, tx, ty = self.from_cxcy_to_gridxy(cx*W, cy*H, gs, (W, H))
                if gx >= gs or gy >= gs: continue
                for a in range(anchors.shape[1]):
                    aw, ah = anchors[s, a]
                    iou = self._compute_iou_wh(w, h, aw, ah)
                    # (YOLOv5/v8  0.2-0.3 
                    if iou > 0.3: 
                        target_obj[s][gy, gx, a] = 1.0
                        target_boxes[s][gy, gx, a] = torch.tensor([tx, ty, w, h])
                        target_cls[s][gy, gx, a, :] = smooth / (self.num_classes - 1)
                        target_cls[s][gy, gx, a, int(label)] = 1.0 - smooth

        final_target = []
        for i in range(num_scales):
            final_target.append(torch.cat(
                (target_boxes[i], target_obj[i].unsqueeze(-1), target_cls[i]),
                dim=-1
            ))
        return final_target

    def load_image_raw(self, idx):
        # (12:44 AM 
        fname = self.fnames[idx]
        image_path = os.path.join(self.root, fname) 
        img = cv2.imread(image_path)
        
        if img is None:
            raise FileNotFoundError(f"Failed to load image at path: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = np.array(self.boxes[idx], dtype=np.float32) if self.boxes else np.empty((0, 4), dtype=np.float32)
        labels = np.array(self.labels[idx], dtype=np.int64) if self.labels else np.empty((0,), dtype=np.int64)
        
        return img, boxes, labels

    def load_mosaic(self, indices):
        # ( 12:41 PM  SOTA Mosaic 
        s = self.image_size
        labels4 = []
        boxes4  = []
        mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        yc = int(random.uniform(s // 2, s + s // 2))
        xc = int(random.uniform(s // 2, s + s // 2))

        for i, index in enumerate(indices):
            img, boxes, labels = self.load_image_raw(index)
            h0, w0 = img.shape[:2]

            scale = random.uniform(0.5, 1.5)
            r = min((s * scale) / h0, (s * scale) / w0)
            new_h, new_w = int(h0 * r), int(w0 * r)
            if new_h <= 0 or new_w <= 0:
                continue
            img_resz = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = xc - new_w, yc - new_h, xc, yc
                x1b, y1b, x2b, y2b = 0, 0, new_w, new_h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, yc - new_h, xc + new_w, yc
                x1b, y1b, x2b, y2b = 0, 0, new_w, new_h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = xc - new_w, yc, xc, yc + new_h
                x1b, y1b, x2b, y2b = 0, 0, new_w, new_h
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, xc + new_w, yc + new_h
                x1b, y1b, x2b, y2b = 0, 0, new_w, new_h

            x1a_clip = max(x1a, 0)
            y1a_clip = max(y1a, 0)
            x2a_clip = min(x2a, s * 2)
            y2a_clip = min(y2a, s * 2)

            paste_w = max(x2a_clip - x1a_clip, 0)
            paste_h = max(y2a_clip - y1a_clip, 0)
            if paste_w == 0 or paste_h == 0:
                continue

            x1b_adj = x1b + (x1a_clip - x1a)
            y1b_adj = y1b + (y1a_clip - y1a)
            x2b_adj = x1b_adj + paste_w
            y2b_adj = y1b_adj + paste_h

            x1b_adj = int(np.clip(x1b_adj, 0, new_w))
            y1b_adj = int(np.clip(y1b_adj, 0, new_w))
            x2b_adj = int(np.clip(x2b_adj, 0, new_w))
            y2b_adj = int(np.clip(y2b_adj, 0, new_w))

            paste_w = x2b_adj - x1b_adj
            paste_h = y2b_adj - y1b_adj
            if paste_w == 0 or paste_h == 0:
                continue
            x2a_clip = x1a_clip + paste_w
            y2a_clip = y1a_clip + paste_h

            mosaic_img[y1a_clip:y2a_clip, x1a_clip:x2a_clip] = img_resz[y1b_adj:y2b_adj, x1b_adj:x2b_adj]

            if len(boxes):
                boxes = np.array(boxes, dtype=np.float32).copy()
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / w0)
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / h0)
                
                dx = x1a_clip - x1b_adj
                dy = y1a_clip - y1b_adj
                boxes[:, [0, 2]] += dx
                boxes[:, [1, 3]] += dy
                boxes4.append(boxes)
                labels4.append(np.array(labels, dtype=np.int64))

        if len(boxes4):
            boxes4 = np.concatenate(boxes4, 0)
            labels4 = np.concatenate(labels4, 0)
        else:
            boxes4 = np.zeros((0, 4), dtype=np.float32)
            labels4 = np.zeros((0,), dtype=np.int64)

        x1 = xc - s // 2
        y1 = yc - s // 2
        x2 = x1 + s
        y2 = y1 + s
        mosaic_img = mosaic_img[y1:y2, x1:x2]

        if len(boxes4):
            boxes4[:, [0, 2]] = np.clip(boxes4[:, [0, 2]] - x1, 0, s)
            boxes4[:, [1, 3]] = np.clip(boxes4[:, [1, 3]] - y1, 0, s)
            w = boxes4[:, 2] - boxes4[:, 0]
            h = boxes4[:, 3] - boxes4[:, 1]
            keep = (w > 2) & (h > 2)
            boxes4 = boxes4[keep]
            labels4 = labels4[keep]

        return mosaic_img, boxes4.tolist(), labels4.tolist()


    def __getitem__(self, idx):
        # (12:51 AM 
        use_mosaic = self.train and self.enable_mosaic and random.random() < self.mosaic_prob
        
        if use_mosaic:
            indices = [idx] + [random.randint(0, self.num_samples - 1) for _ in range(3)]
            try:
                img, boxes, labels = self.load_mosaic(indices) 
                if len(boxes) == 0:
                    use_mosaic = False
            except Exception as e:
                # print(f"Error in load_mosaic for index {idx}: {e}. Falling back.")
                use_mosaic = False
        
        if not use_mosaic:
            try:
                img, boxes_np, labels_np = self.load_image_raw(idx)
            except Exception as e:
                # print(f"Failed to load_image_raw for {idx}: {e}. Retrying.")
                return self.__getitem__(random.randint(0, self.num_samples - 1))
                
            boxes = list(boxes_np) 
            labels = list(labels_np)
            
            if self.train:
                t = self.crop_transform(image=img, bboxes=boxes, cls_labels=labels)
            else:
                t = self.resize_transform(image=img, bboxes=boxes, cls_labels=labels)
            
            img, boxes, labels = t['image'], t['bboxes'], t['cls_labels']

        try:
            t = self.augment(image=img, bboxes=boxes, cls_labels=labels)
            image_t, bboxes_t, labels_t = t['image'], t['bboxes'], t['cls_labels']
            
            if len(bboxes_t) == 0:
                 raise ValueError("Augmentations removed all boxes.")
                 
        except Exception as e:
            t = self.test_pipe(image=img, bboxes=boxes, cls_labels=labels)
            image_t, bboxes_t, labels_t = t['image'], t['bboxes'], t['cls_labels']

        if self.encode_target:
            if len(bboxes_t) == 0:
                return self.__getitem__(random.randint(0, self.num_samples - 1))
            
            target = self.encoder(image_t, bboxes_t, labels_t)
            
            if target is None: 
                return self.__getitem__(random.randint(0, self.num_samples - 1))
                
            return image_t, target
        else:
            return image_t, bboxes_t, labels_t


    def __len__(self):
        return len(self.fnames)

def collate_fn(batch):
    # ( 03:48 
    images = []
    if len(batch[0]) == 2:
        targets_list = [[] for _ in range(len(GRID_SIZES))]
        for image, target in batch:
            images.append(image)
            for i in range(len(GRID_SIZES)):
                targets_list[i].append(target[i])
        targets = [torch.stack(t, 0) for t in targets_list]
        return torch.stack(images, 0), targets
    elif len(batch[0]) == 3:
        target_list = []
        for image, boxes, labels in batch:
            images.append(image)
            instances = []
            for box, label in zip(boxes, labels):
                instances.append(list(box) + [int(label)])
            target_list.append(instances)
        return torch.stack(images, 0), target_list
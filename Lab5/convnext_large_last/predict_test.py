import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.config_new import VOC_CLASSES, YOLO_IMG_DIM
from src.dataset_mosaic_last import test_data_pipelines
# 1. (修改) 確保從 yolo.py 匯入 NMS 函式
from src.yolo_convnext_large import getODmodel, non_max_suppression


def parse_args() -> argparse.Namespace:
    # (您 11:28 AM 的版本，很好，保持不變)
    parser = argparse.ArgumentParser(description="Run YOLOv3 inference on the Kaggle test split.")
    parser.add_argument("--weights", type=Path, default=Path("checkpoints/best_detector.pth"),
                        help="Path to the trained model weights.")
    parser.add_argument("--test-list", type=Path, default=Path("dataset/vocall_test.txt"),
                        help="File containing test image names (one per line).")
    parser.add_argument("--images-dir", type=Path, default=Path("dataset/image"),
                        help="Directory containing the test images.")
    parser.add_argument("--output", type=Path, default=Path("result.csv"),
                        help="Where to write the prediction CSV (matching Kaggle result format).")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--nms-thres", type=float, default=0.4, help="IOU threshold for NMS.")
    parser.add_argument("--tta", type=str, default="hflip", choices=["none", "hflip", "hvflip"],
                        help="Test-time augmentation mode.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (cuda, cpu, or auto).")
    return parser.parse_args()


def build_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    # (保持不變)
    model = getODmodel(pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path) -> Tuple[torch.Tensor, Tuple[int, int]]:
    # (保持不變)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = test_data_pipelines(image=image_rgb, bboxes=[], cls_labels=[])
    tensor = transformed["image"]
    return tensor, (orig_w, orig_h)


def convert_detections(
    detections: torch.Tensor,
    orig_size: Tuple[int, int],
    input_dim: int = YOLO_IMG_DIM,
) -> List[List]:
    # (您 11:28 AM 的版本，很好，保持不變)
    """把 (cx,cy,w,h) normalized 轉回原圖座標（Kaggle CSV 格式）"""
    orig_w, orig_h = orig_size
    sx = orig_w / float(input_dim)
    sy = orig_h / float(input_dim)

    boxes = []
    for detection in detections:
        x_center, y_center, width, height, obj_conf, cls_conf, cls_idx = detection.tolist()
        prob = float(obj_conf * cls_conf)
        if prob <= 0:
            continue

        # 先在輸入空間算像素框
        x1_i = (x_center - width / 2.0) * input_dim
        y1_i = (y_center - height / 2.0) * input_dim
        x2_i = (x_center + width / 2.0) * input_dim
        y2_i = (y_center + height / 2.0) * input_dim

        # 再縮放回原圖
        x1 = max(0.0, min(x1_i * sx, orig_w - 1))
        y1 = max(0.0, min(y1_i * sy, orig_h - 1))
        x2 = max(0.0, min(x2_i * sx, orig_w - 1))
        y2 = max(0.0, min(y2_i * sy, orig_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(cls_idx)
        class_name = VOC_CLASSES[class_id]
        boxes.append([class_name, round(prob, 6), x1, y1, x2, y2])
    return boxes


# -----------------------
# TTA & Merge Utilities
# -----------------------

# 2. (刪除) 
# 刪掉你原本的 _xywhn_to_xyxy_pixels, nms_after_merge, tta_infer 
# 替換成下面這一個 process_batch 函式

# -----------------------
# Inference Loop
# -----------------------

def run_inference(args: argparse.Namespace) -> None:
    # (您 11:28 AM 的版本，很好，保持不變)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Running inference on {device}, TTA mode: {args.tta}") # (我幫您改了 log)

    image_dir = args.images_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    with args.test_list.open() as f:
        image_names = [line.strip() for line in f if line.strip()]

    model = build_model(args.weights, device)

    results = []
    batch_tensors: List[torch.Tensor] = []
    batch_meta: List[Tuple[str, Tuple[int, int]]] = []

    for name in image_names:
        image_path = image_dir / name
        tensor, orig_size = preprocess_image(image_path)
        batch_tensors.append(tensor)
        batch_meta.append((name, orig_size))

        if len(batch_tensors) == args.batch_size:
            results.extend(
                process_batch(
                    model,
                    batch_tensors,
                    batch_meta,
                    device,
                    args.conf_thres,
                    args.nms_thres,
                    args.tta, # 傳入 TTA 模式
                )
            )
            batch_tensors.clear()
            batch_meta.clear()

    if batch_tensors:
        results.extend(
            process_batch(
                model,
                batch_tensors,
                batch_meta,
                device,
                args.conf_thres,
                args.nms_thres,
                args.tta, # 傳入 TTA 模式
            )
        )

    write_predictions(args.output, results)
    print(f"Predictions written to {args.output}") # (我幫您加了 log)


def process_batch(
    model: torch.nn.Module,
    batch_tensors: List[torch.Tensor],
    batch_meta: List[Tuple[str, Tuple[int, int]]],
    device: torch.device,
    conf_thres: float,
    nms_thres: float,
    tta_mode: str,
) -> List[Tuple[str, str]]:
    """
    (完整修正版) 
    處理一個批次，根據 tta_mode 決定推論策略。
    """
    images = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        if tta_mode == "none":
            # 模式 1: 標準推論 (呼叫內建 NMS)
            detections = model.inference(images, conf_thres=conf_thres, nms_thres=nms_thres)
        
        else:
            # 模式 2: TTA (手動 NMS)
            # 1. 取得原圖 pre-NMS 框
            pre_nms_orig = model.get_pre_nms_boxes(images) # (B, N, 5+C)
            
            # 2. 取得 TTA 圖 pre-NMS 框
            all_pre_nms_boxes = [pre_nms_orig]

            if tta_mode in ("hflip", "hvflip"):
                # 2.1 水平翻轉
                pre_nms_hflip = model.get_pre_nms_boxes(torch.flip(images, dims=[3]))
                pre_nms_hflip[..., 0] = 1.0 - pre_nms_hflip[..., 0] # Un-flip cx
                all_pre_nms_boxes.append(pre_nms_hflip)

            if tta_mode == "hvflip":
                # 2.2 垂直翻轉
                pre_nms_vflip = model.get_pre_nms_boxes(torch.flip(images, dims=[2]))
                pre_nms_vflip[..., 1] = 1.0 - pre_nms_vflip[..., 1] # Un-flip cy
                all_pre_nms_boxes.append(pre_nms_vflip)

            # 3. 合併所有 TTA 來源的框
            # (B, N * TTA_Count, 5+C)
            combined_boxes = torch.cat(all_pre_nms_boxes, dim=1)

            # 4. 只跑一次 NMS (呼叫 src.yolo 裡的函式)
            detections = non_max_suppression(
                combined_boxes, 
                conf_thres=conf_thres, 
                nms_thres=nms_thres
            ) # 回傳 List[Tensor]

    # (格式化輸出，保持不變)
    batch_results: List[Tuple[str, str]] = []
    for det, (name, orig_size) in zip(detections, batch_meta):
        if det is None or len(det) == 0:
            prediction_list = "[]"
        else:
            boxes = convert_detections(det.cpu(), orig_size)
            prediction_list = str(boxes)
        batch_results.append((name, prediction_list))
    return batch_results


def write_predictions(output_path: Path, predictions: List[Tuple[str, str]]) -> None:
    # (保持不變)
    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "prediction_list"])
        for image_name, prediction_list in predictions:
            writer.writerow([image_name, prediction_list])


if __name__ == "__main__":
    run_inference(parse_args())
"""
RC카 보행자 탐지용 커스텀 디텍터 학습 스크립트.

데이터셋: archive/{train,valid,test}/{같은 이름}/ 아래 이미지 + _annotations.csv
          (filename,width,height,class,xmin,ymin,xmax,ymax / pascal_voc 절대좌표)

핵심 수정 사항 (원본 노트북 대비):
  - build_target / loss / decode의 wh 인코딩을 log(bw/W) 기준으로 통일.
    (원본은 target에 bw/W를 저장하고 loss는 sigmoid(pred)와 비교했는데,
     추론 decode는 exp(pred)*W로 복원해서 학습-추론 수식이 어긋났음.
     그 결과 박스 크기가 신뢰할 수 없게 나와 몸통/얼굴이 따로 잡히는 원인이 됨.
     지금은 loss도 raw pred와 log(bw/W)를 그대로 비교하므로 decode의 exp가
     정확한 역함수가 됨)
  - PersonDataset이 매 __getitem__마다 전체 DataFrame을 스캔하지 않도록
    파일명 -> 박스 배열 dict를 미리 그룹핑.
  - mAP@0.5 평가지표 도입 (기존엔 conf=0.25 한 점에서의 recall만 측정).
  - ImageNet 사전학습 MobileNetV3-Small 백본 옵션 (--backbone mobilenet),
    사전학습 백본에는 차등 학습률(backbone lr * --backbone-lr-mult) 적용.
    mobilenet 백본일 때만 입력에 ImageNet 정규화(mean/std) 적용 (custom은 0~1 유지).
  - TorchScript 변환 시 example 입력 shape 버그 수정 (H, W 순서).
  - [2단계] 다중 양성 셀 할당: GT박스 중심 셀 + 인접 최대 2셀(가로/세로 각 1개)에 동시에
    정답 할당 (YOLOv5 스타일). 오프셋 범위가 [0,1) -> [-0.5,1.5)로 확장되어 loss/decode
    모두 sigmoid(x)*2-0.5 사용 (detect_video.py도 동일하게 맞춰야 함).
  - [2단계] RandomSizedBBoxSafeCrop으로 스케일 증강 추가 (클로즈업/큰 스케일 사람 비중 증가).
"""
import argparse
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A

GRID_STRIDE = 16

# mobilenet 백본은 ImageNet 사전학습 가중치를 쓰므로 그 학습 당시 분포(정규화)를 맞춰야
# 사전학습 feature가 제대로 작동함. custom 백본은 처음부터 랜덤 초기화라 0~1 그대로 사용.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------
# 데이터셋
# --------------------------------------------------------------------------
def split_paths(data_root, split):
    d = os.path.join(data_root, split, split)
    return d, os.path.join(d, "_annotations.csv")


def build_train_transform(img_w, img_h):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            # 클로즈업(큰 스케일) 사람 비중을 늘리기 위한 확대 크롭: 박스를 보존하는 랜덤
            # 크기로 자른 뒤 원래 해상도로 다시 리사이즈 -> 사람이 화면에서 차지하는 비율이 커짐
            A.RandomSizedBBoxSafeCrop(height=img_h, width=img_w, erosion_rate=0.0, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-5, 5), p=0.3),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 20), hole_width_range=(8, 20), p=0.2),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


class PersonDataset(Dataset):
    def __init__(self, image_dir, csv_file, img_w, img_h, max_images=None, transform=None, normalize=False):
        self.image_dir = image_dir
        self.img_w = img_w
        self.img_h = img_h
        self.transform = transform
        self.normalize = normalize

        df = pd.read_csv(csv_file)
        df = df[df["class"] == "person"]

        # 파일명별로 미리 그룹핑 -> __getitem__마다 O(N) 스캔하는 것을 방지
        self.boxes_by_file = {
            fname: g[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
            for fname, g in df.groupby("filename")
        }
        self.image_list = sorted(self.boxes_by_file.keys())

        if max_images is not None:
            random.seed(32)
            self.image_list = random.sample(self.image_list, min(max_images, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        filename = self.image_list[idx]
        boxes = self.boxes_by_file[filename].copy()

        path = os.path.join(self.image_dir, filename)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes = boxes[valid]

        # 원본(최대 수천px) 대신 320x240으로 먼저 줄인 뒤 augmentation을 적용함.
        # 순서를 바꾸면 (1) augmentation 비용이 원본 해상도에 비례해 커지는 문제가 없어지고,
        # (2) CoarseDropout의 절대 픽셀(8~20px) 구멍 크기가 최종 학습 해상도 기준으로
        #     의도한 만큼(작은 가림)의 비중을 가지게 됨 (원본 대형 이미지 기준이면 사실상 안 보이는 크기였음)
        h, w = image.shape[:2]
        sx, sy = self.img_w / w, self.img_h / h
        image = cv2.resize(image, (self.img_w, self.img_h))
        if len(boxes) > 0:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * sx, 0, self.img_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * sy, 0, self.img_h)

        if self.transform is not None and len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, labels=np.ones(len(boxes)))
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32)

        image = image.astype(np.float32) / 255.0
        if self.normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(boxes, dtype=torch.float32)


def collate_fn(batch):
    images = torch.stack([img for img, _ in batch], dim=0)
    boxes_list = [boxes for _, boxes in batch]
    return images, boxes_list


# --------------------------------------------------------------------------
# 타겟 생성 (그리드 좌표 인코딩)
# --------------------------------------------------------------------------
def get_grid_size(img_w, img_h):
    return img_w // GRID_STRIDE, img_h // GRID_STRIDE


def build_target(boxes, grid_w, grid_h, img_w, img_h):
    """다중 양성 셀 할당(YOLOv5 스타일): 중심 셀 + 인접 최대 2셀(가로/세로 각 1개, 박스
    중심이 셀의 어느 반쪽에 있는지로 결정)에 동시에 정답을 할당함. 양성 샘플 수가
    늘어나 recall이 개선됨. 인접 셀은 자기 기준 오프셋이 [0,1) 밖(약 -0.5~1.5)으로
    나갈 수 있어서, decode/loss 쪽도 sigmoid(x)*2-0.5로 범위를 맞춰야 함 (decode_prediction 참고)."""
    target = torch.zeros((5, grid_h, grid_w), dtype=torch.float32)
    occupied = torch.zeros((grid_h, grid_w), dtype=torch.bool)

    cell_w = img_w / grid_w
    cell_h = img_h / grid_h

    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 1 or bh <= 1:
            continue

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        gx_f = cx / cell_w
        gy_f = cy / cell_h
        gx, gy = int(gx_f), int(gy_f)
        if gx < 0 or gy < 0 or gx >= grid_w or gy >= grid_h:
            continue

        frac_x = gx_f - gx
        frac_y = gy_f - gy

        cells = [(gx, gy)]
        if frac_x < 0.5 and gx - 1 >= 0:
            cells.append((gx - 1, gy))
        elif frac_x >= 0.5 and gx + 1 < grid_w:
            cells.append((gx + 1, gy))
        if frac_y < 0.5 and gy - 1 >= 0:
            cells.append((gx, gy - 1))
        elif frac_y >= 0.5 and gy + 1 < grid_h:
            cells.append((gx, gy + 1))

        for cgx, cgy in cells:
            if occupied[cgy, cgx]:
                continue
            occupied[cgy, cgx] = True

            target[0, cgy, cgx] = 1.0
            target[1, cgy, cgx] = gx_f - cgx
            target[2, cgy, cgx] = gy_f - cgy
            # log 인코딩: decode의 exp(pred)*W와 정확히 역함수 관계가 되도록 함
            target[3, cgy, cgx] = math.log(bw / img_w)
            target[4, cgy, cgx] = math.log(bh / img_h)

    return target


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------
class DSConv(nn.Module):
    """Depthwise Separable Conv: 엣지 디바이스(RPi4)를 겨냥한 경량 conv 블록."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out, inplace=True)


class CustomBackbone(nn.Module):
    """DSConv + ResidualBlock, stride 16 (320x240 -> 20x15), 256채널 출력."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(DSConv(32, 64, 2), ResidualBlock(64))
        self.layer2 = nn.Sequential(DSConv(64, 128, 2), ResidualBlock(128))
        self.layer3 = nn.Sequential(DSConv(128, 256, 2), ResidualBlock(256))
        self.layer4 = nn.Sequential(DSConv(256, 256, 1), ResidualBlock(256))
        self.out_channels = 256

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def _download_with_certifi_fallback(download_fn):
    """torch.hub 다운로드는 stdlib urllib을 쓰기 때문에 pip-system-certs(=requests만 패치)로는
    고쳐지지 않는 Windows 인증서 저장소 파싱 버그(ASN1: NOT_ENOUGH_DATA)가 있음.
    실패 시 certifi CA 번들로 HTTPS 컨텍스트를 한 번 교체해서 재시도한다."""
    try:
        return download_fn()
    except Exception as e:  # noqa: BLE001 - SSL/네트워크 오류를 폭넓게 흡수 후 한 번 더 시도
        try:
            import ssl
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
            ssl._create_default_https_context = lambda: ctx
            print(f"[안내] 기본 SSL 컨텍스트 실패({e!r}) -> certifi CA 번들로 재시도합니다.")
            return download_fn()
        except Exception:
            raise e


class MobileNetBackbone(nn.Module):
    """ImageNet 사전학습 MobileNetV3-Small의 앞 5개 블록만 사용 (stride 16, 40채널)."""

    def __init__(self, pretrained=True, pretrained_path=None):
        super().__init__()
        import torchvision
        from torchvision.models import MobileNet_V3_Small_Weights

        base = None
        if pretrained_path is not None:
            base = torchvision.models.mobilenet_v3_small(weights=None)
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            base.load_state_dict(state)
        elif pretrained:
            try:
                base = _download_with_certifi_fallback(
                    lambda: torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                )
            except Exception as e:  # noqa: BLE001 - 네트워크/인증서 문제 등 환경 이슈를 폭넓게 흡수
                print(f"[경고] ImageNet 사전학습 가중치 다운로드 실패 ({e!r}).")
                print("       --pretrained-path로 로컬 가중치를 지정하거나 랜덤 초기화로 계속 진행합니다.")
        if base is None:
            base = torchvision.models.mobilenet_v3_small(weights=None)

        # features[0:5] -> 240x320 입력 기준 (40, 15, 20) : stride 16과 정확히 일치
        self.features = base.features[:5]
        self.out_channels = 40

    def forward(self, x):
        return self.features(x)


class PersonDetector(nn.Module):
    def __init__(self, backbone="custom", pretrained=True, pretrained_path=None):
        super().__init__()
        if backbone == "custom":
            self.backbone = CustomBackbone()
        elif backbone == "mobilenet":
            self.backbone = MobileNetBackbone(pretrained=pretrained, pretrained_path=pretrained_path)
        else:
            raise ValueError(f"알 수 없는 backbone: {backbone}")

        backbone_ch = self.backbone.out_channels
        # custom backbone은 이미 256채널이라 neck이 항등함수가 되어 기존 head와 완전히 동일하게 유지됨
        if backbone_ch != 256:
            self.neck = nn.Sequential(
                nn.Conv2d(backbone_ch, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        else:
            self.neck = nn.Identity()

        self.head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            DSConv(64, 64, stride=1),
            nn.Conv2d(64, 5, kernel_size=1),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p = torch.sigmoid(pred)
        pt = torch.where(target == 1, p, 1 - p)
        alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
        loss = alpha_factor * (1 - pt).pow(self.gamma) * bce
        return loss.sum()


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_loss = FocalLoss()
        self.box_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, pred, target):
        obj_pred = pred[:, 0]
        obj_gt = target[:, 0]

        obj_loss_sum = self.obj_loss(obj_pred, obj_gt)

        mask = obj_gt > 0
        num_positives = torch.clamp(mask.sum().float(), min=1.0)

        if mask.any():
            pred_box = pred[:, 1:5].permute(0, 2, 3, 1)[mask]
            gt_box = target[:, 1:5].permute(0, 2, 3, 1)[mask]

            # 다중 양성 셀 할당으로 target 오프셋 범위가 [0,1) -> [-0.5,1.5)로 확장됐으므로
            # sigmoid(x)*2-0.5로 맞춤 (decode_prediction과 반드시 동일해야 함)
            pred_xy = torch.sigmoid(pred_box[:, 0:2]) * 2.0 - 0.5
            # wh는 sigmoid를 거치지 않은 raw 값 그대로 사용
            # (target이 log(bw/W) 인코딩이고, decode도 exp(raw_pred)*W이므로 이래야 학습/추론이 일치)
            pred_wh = pred_box[:, 2:4]
            pred_box_processed = torch.cat([pred_xy, pred_wh], dim=-1)

            box_l1_loss = self.box_loss(pred_box_processed, gt_box)
        else:
            box_l1_loss = torch.tensor(0.0, device=pred.device)

        return (obj_loss_sum / num_positives) + (10.0 * box_l1_loss / num_positives)


# --------------------------------------------------------------------------
# Decode / NMS / IoU
# --------------------------------------------------------------------------
WH_LOG_CLAMP = 4.0  # exp() 오버플로 방지용 (초기/불안정 학습 시 raw pred가 튈 수 있음)


def decode_prediction(pred, img_w, img_h, conf_threshold=0.25):
    boxes = []
    pred = pred.detach().cpu()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    cell_w = img_w / grid_w
    cell_h = img_h / grid_h

    obj = torch.sigmoid(pred[0])
    ys, xs = torch.where(obj > conf_threshold)

    for y, x in zip(ys.tolist(), xs.tolist()):
        # 다중 양성 셀 할당으로 오프셋 범위가 [-0.5,1.5)까지 확장됨 (build_target 참고)
        tx = (torch.sigmoid(pred[1, y, x]) * 2.0 - 0.5).item()
        ty = (torch.sigmoid(pred[2, y, x]) * 2.0 - 0.5).item()
        tw = torch.clamp(pred[3, y, x], max=WH_LOG_CLAMP).item()
        th = torch.clamp(pred[4, y, x], max=WH_LOG_CLAMP).item()

        bw = math.exp(tw) * img_w
        bh = math.exp(th) * img_h

        cx = (x + tx) * cell_w
        cy = (y + ty) * cell_h

        xmin = max(0.0, cx - bw / 2)
        ymin = max(0.0, cy - bh / 2)
        xmax = min(float(img_w), cx + bw / 2)
        ymax = min(float(img_h), cy + bh / 2)

        boxes.append([xmin, ymin, xmax, ymax, obj[y, x].item()])

    return boxes


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def nms(boxes, iou_threshold=0.3):
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if calculate_iou(best[:4], b[:4]) < iou_threshold]

    return keep


# --------------------------------------------------------------------------
# mAP@IoU 계산 (VOC2012 all-point 방식, 클래스 1개)
# --------------------------------------------------------------------------
def compute_ap(all_predictions, all_gts, iou_threshold=0.5):
    total_gt = sum(len(v) for v in all_gts.values())
    if total_gt == 0:
        return 0.0
    if not all_predictions:
        return 0.0

    matched = {img_id: [False] * len(boxes) for img_id, boxes in all_gts.items()}
    preds_sorted = sorted(all_predictions, key=lambda p: p[1], reverse=True)

    tp = np.zeros(len(preds_sorted), dtype=np.float64)
    fp = np.zeros(len(preds_sorted), dtype=np.float64)

    for i, (img_id, _conf, box) in enumerate(preds_sorted):
        gts = all_gts.get(img_id, [])
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched[img_id][j]:
                continue
            iou = calculate_iou(box, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_j >= 0 and best_iou >= iou_threshold:
            tp[i] = 1.0
            matched[img_id][best_j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# --------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------
def evaluate(model, loader, criterion, device, img_w, img_h, grid_w, grid_h,
             conf_threshold=0.25, recall_iou=0.3, nms_iou=0.3, map_iou=0.5):
    model.eval()

    total_loss = 0.0
    detected_objects = 0
    total_objects = 0
    total_predictions = 0

    all_predictions = []
    all_gts = {}
    img_id = 0

    with torch.no_grad():
        for images, boxes_list in loader:
            images = images.to(device)
            targets = torch.stack(
                [build_target(b, grid_w, grid_h, img_w, img_h) for b in boxes_list]
            ).to(device)

            pred = model(images)
            loss = criterion(pred, targets)
            total_loss += loss.item()

            for i in range(len(images)):
                pred_boxes = decode_prediction(pred[i], img_w, img_h, conf_threshold)
                pred_boxes = nms(pred_boxes, nms_iou)

                gt_boxes = boxes_list[i].tolist()
                all_gts[img_id] = gt_boxes
                for box in pred_boxes:
                    all_predictions.append((img_id, box[4], box[:4]))

                total_predictions += len(pred_boxes)
                total_objects += len(gt_boxes)
                for gt in gt_boxes:
                    if any(calculate_iou(pb[:4], gt) >= recall_iou for pb in pred_boxes):
                        detected_objects += 1

                img_id += 1

    avg_loss = total_loss / len(loader)
    recall = detected_objects / total_objects if total_objects > 0 else 0.0
    avg_pred = total_predictions / len(loader.dataset)
    mAP = compute_ap(all_predictions, all_gts, iou_threshold=map_iou)

    return {"loss": avg_loss, "recall": recall, "avg_pred": avg_pred, "mAP50": mAP}


# --------------------------------------------------------------------------
# 학습
# --------------------------------------------------------------------------
def build_optimizer(model, args):
    if args.backbone == "mobilenet" and args.backbone_lr_mult != 1.0:
        backbone_params = list(model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        param_groups = [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": other_params, "lr": args.lr},
        ]
    else:
        param_groups = model.parameters()

    return torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def plot_history(train_losses, val_losses, val_maps, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(val_maps, label="Validation mAP@0.5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close(fig)


def export_torchscript(model, img_w, img_h, device, output_path):
    model.eval()
    # [버그 수정] 원본은 (1,3,W,H)로 잘못 생성했음. 모델은 (B,C,H,W)를 기대함
    example = torch.randn(1, 3, img_h, img_w, device=device)
    scripted = torch.jit.trace(model, example)
    scripted.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="RC카 보행자 탐지 모델 학습")
    parser.add_argument("--data-root", default="./archive")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--backbone", choices=["custom", "mobilenet"], default="custom")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--pretrained-path", default=None, help="로컬에 저장된 mobilenet_v3_small state_dict 경로")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)

    parser.add_argument("--img-width", type=int, default=320)
    parser.add_argument("--img-height", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--recall-iou", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--map-iou", type=float, default=0.5)

    parser.add_argument("--max-train-images", type=int, default=None, help="빠른 스모크 테스트용 서브셋 크기")
    parser.add_argument("--max-val-images", type=int, default=None)

    parser.add_argument("--resume", default=None, help="이어서 학습할 checkpoint(.pth) 경로")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-export", dest="export", action="store_false", default=True)
    args = parser.parse_args()

    if args.img_width % GRID_STRIDE != 0 or args.img_height % GRID_STRIDE != 0:
        raise ValueError(f"img-width/height는 {GRID_STRIDE}의 배수여야 합니다.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("DEVICE:", device)

    grid_w, grid_h = get_grid_size(args.img_width, args.img_height)
    print(f"Grid size: {grid_w}x{grid_h} (stride {GRID_STRIDE})")

    # ---------------- 데이터 ----------------
    train_dir, train_csv = split_paths(args.data_root, "train")
    valid_dir, valid_csv = split_paths(args.data_root, "valid")

    normalize = args.backbone == "mobilenet"
    train_dataset = PersonDataset(
        train_dir, train_csv, args.img_width, args.img_height,
        max_images=args.max_train_images, transform=build_train_transform(args.img_width, args.img_height), normalize=normalize,
    )
    val_dataset = PersonDataset(
        valid_dir, valid_csv, args.img_width, args.img_height,
        max_images=args.max_val_images, transform=None, normalize=normalize,
    )
    print(f"Train: {len(train_dataset)}  Valid: {len(val_dataset)}")

    # persistent_workers=True: 매 epoch(= DataLoader.__iter__ 호출)마다 worker 프로세스를
    # 새로 spawn/재import하는 오버헤드를 없앰 (Windows spawn 방식에서 특히 비용이 큼)
    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
    )

    # ---------------- 모델 ----------------
    model = PersonDetector(
        backbone=args.backbone, pretrained=args.pretrained, pretrained_path=args.pretrained_path
    ).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, 3, args.img_height, args.img_width, device=device)
        out_shape = model(dummy).shape
    assert out_shape[2] == grid_h and out_shape[3] == grid_w, (
        f"모델 출력 그리드({out_shape[2]}x{out_shape[3]})가 기대값({grid_h}x{grid_w})과 다릅니다."
    )

    criterion = DetectionLoss()
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    train_losses, val_losses, val_maps, val_recalls = [], [], [], []
    start_epoch = 0
    best_map = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        train_losses = ckpt.get("train_loss", [])
        val_losses = ckpt.get("val_loss", [])
        val_maps = ckpt.get("val_map", [])
        best_map = max(val_maps) if val_maps else 0.0
        print(f"체크포인트에서 재개: {args.resume} (epoch {start_epoch}부터)")

    best_path = os.path.join(args.output_dir, "best_model.pth")
    last_path = os.path.join(args.output_dir, "person_detector.pth")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        # 파일로 리다이렉트된 백그라운드 실행 시 tqdm의 \r 갱신이 로그를 어지럽히지 않도록 비활성화
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True,
            disable=not sys.stdout.isatty(),
        )

        for images, boxes_list in progress_bar:
            images = images.to(device)
            targets = torch.stack(
                [build_target(b, grid_w, grid_h, args.img_width, args.img_height) for b in boxes_list]
            ).to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(images)
                loss = criterion(pred, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        val_metrics = evaluate(
            model, valid_loader, criterion, device, args.img_width, args.img_height, grid_w, grid_h,
            conf_threshold=args.conf_threshold, recall_iou=args.recall_iou,
            nms_iou=args.nms_iou, map_iou=args.map_iou,
        )
        scheduler.step(val_metrics["loss"])

        val_losses.append(val_metrics["loss"])
        val_maps.append(val_metrics["mAP50"])
        val_recalls.append(val_metrics["recall"])

        if val_metrics["mAP50"] > best_map:
            best_map = val_metrics["mAP50"]
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {avg_loss:.4f}  Val Loss: {val_metrics['loss']:.4f}  "
            f"Val mAP@0.5: {val_metrics['mAP50']:.4f}  Val Recall: {val_metrics['recall']:.4f}  "
            f"(best mAP: {best_map:.4f})"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_map": val_maps,
                "args": vars(args),
            },
            last_path,
        )

    try:
        plot_history(train_losses, val_losses, val_maps, args.output_dir)
    except Exception as e:  # noqa: BLE001 - 그래프 저장 실패가 학습 결과 자체를 무효화하진 않음
        print(f"[경고] 학습 곡선 저장 실패: {e!r}")

    if args.export:
        export_path = os.path.join(args.output_dir, "person_detector_script.pt")
        export_torchscript(model, args.img_width, args.img_height, device, export_path)
        print(f"TorchScript 저장 완료: {export_path}")

    print(f"학습 종료. best mAP@0.5 = {best_map:.4f} ({best_path})")


if __name__ == "__main__":
    main()

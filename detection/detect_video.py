"""
보행자 탐지 모델(TorchScript)로 웹캠/영상을 추론하는 스크립트.
YOLOv8 계열과, hi/01_custom_model의 커스텀 grid 계열 모델을 둘 다 지원한다
(--model-type으로 선택. 기본값 yolo는 기존 동작 그대로 유지).

RPi4 배포를 염두에 두고 torch, cv2, numpy만 있으면 동작함
(pandas/albumentations 등 학습용 의존성 불필요 — requirements.txt 참고).

== --model-type yolo (기본값) ==
letterbox / decode_yolo_output / nms는 viewer/VideoThread.h의 C++ 구현과
반드시 동일한 수식을 유지해야 함. 한쪽만 고치면 같은 모델인데 결과가
달라지므로, 수정 시 양쪽을 같이 볼 것.

모델 출력 형식: model.export(format='torchscript', nms=False)의 결과인
(1, 5, N) = 채널별 [cx, cy, w, h, score].
letterbox 입력 좌표계의 픽셀 단위이고 score는 이미 sigmoid가 적용된
person 클래스 확신도(클래스가 1개라 argmax 불필요).
NMS가 그래프에 포함돼 있지 않으므로 아래 nms()로 직접 처리한다
(nms=True 버전은 torchvision::nms C++ 연산에 의존해서 LibTorch 단독
배포가 안 되기 때문에 raw 출력 + 자체 NMS 조합을 쓴다).

== --model-type grid (hi/01_custom_model 커스텀 모델: 9단계 FPN, 11단계 lite 등) ==
전처리가 letterbox가 아니라 (--img-width, --img-height)로의 단순 resize(종횡비
유지 안 함)이고, 출력도 이미 디코드된 픽셀 좌표가 아니라 grid 원시 출력
(obj_logit, tx_logit, ty_logit, tw_log, th_log)이라 sigmoid/exp로 직접 복원해야
한다. train_person_detector.py의 decode_prediction/decode_levels와 완전히
동일한 수식 - 한쪽만 고치면 같은 모델인데 결과가 달라지므로 수정 시 같이 볼 것.
FPN 모델(9단계 등)은 출력이 텐서 하나가 아니라 레벨별 텐서 tuple이라
decode_levels가 각각 디코드해서 합친다. mobilenet 백본 모델(9단계)은
--normalize를 반드시 켜야 함(ImageNet 정규화로 학습됐음); custom 백본
모델(11단계 등)은 끄면 됨(기본값).
"""
import argparse
import math
import time

import cv2
import numpy as np
import torch

IMGSZ_DEFAULT = 320
PAD_COLOR = 114  # YOLOv8 표준 letterbox 패딩 색

GRID_IMG_WIDTH_DEFAULT = 320
GRID_IMG_HEIGHT_DEFAULT = 240
GRID_WH_LOG_CLAMP = 4.0
# train_person_detector.py의 PersonDataset과 반드시 동일해야 함(mobilenet 백본 전용)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(frame, target_size):
    """종횡비를 유지한 채 target_size 정사각형에 맞추고 남는 곳을 회색으로 패딩.
    박스 좌표를 원본 프레임으로 되돌리려면 scale/pad가 필요해서 같이 반환한다."""
    h, w = frame.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.full((target_size, target_size, 3), PAD_COLOR, dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y


def preprocess(frame, target_size, device):
    canvas, scale, pad_x, pad_y = letterbox(frame, target_size)
    img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor, scale, pad_x, pad_y


def decode_yolo_output(output, conf_threshold):
    """(1,5,N) 또는 (5,N) 출력을 [xmin,ymin,xmax,ymax,score] 리스트로 변환."""
    output = output.detach().cpu()
    if output.dim() == 3:
        output = output[0]

    scores = output[4]
    keep = torch.where(scores > conf_threshold)[0]
    if keep.numel() == 0:
        return []

    sel = output[:, keep]
    cx, cy, bw, bh = sel[0], sel[1], sel[2], sel[3]
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, sel[4]], dim=1)
    return boxes.tolist()


def grid_preprocess(frame, img_w, img_h, device, normalize):
    """--model-type grid 전용 전처리: letterbox 없이 (img_w, img_h)로 단순 resize."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_w, img_h))
    img = img.astype(np.float32) / 255.0
    if normalize:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


def decode_prediction(pred, img_w, img_h, conf_threshold):
    """(5, grid_h, grid_w) raw 출력 -> [xmin,ymin,xmax,ymax,score] 리스트.
    train_person_detector.py::decode_prediction과 동일한 수식."""
    boxes = []
    pred = pred.detach().cpu()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    cell_w = img_w / grid_w
    cell_h = img_h / grid_h

    obj = torch.sigmoid(pred[0])
    ys, xs = torch.where(obj > conf_threshold)

    for y, x in zip(ys.tolist(), xs.tolist()):
        # 다중 양성 셀 할당으로 오프셋 범위가 [-0.5,1.5)까지 확장됨
        tx = (torch.sigmoid(pred[1, y, x]) * 2.0 - 0.5).item()
        ty = (torch.sigmoid(pred[2, y, x]) * 2.0 - 0.5).item()
        tw = torch.clamp(pred[3, y, x], max=GRID_WH_LOG_CLAMP).item()
        th = torch.clamp(pred[4, y, x], max=GRID_WH_LOG_CLAMP).item()

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


def decode_levels(preds, img_w, img_h, conf_threshold):
    """FPN처럼 출력이 레벨별 텐서 tuple/list인 경우 각각 디코드해서 합친다.
    단일 스케일 모델은 preds가 텐서 하나이므로 자동으로 원소 1개짜리로 처리됨."""
    levels = preds if isinstance(preds, (list, tuple)) else (preds,)
    boxes = []
    for pred in levels:
        boxes.extend(decode_prediction(pred, img_w, img_h, conf_threshold))
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


def nms(boxes, iou_threshold=0.45):
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if calculate_iou(best[:4], b[:4]) < iou_threshold]

    return keep


def load_model(weights_path, device):
    """TorchScript 모델 로드. .pt / .torchscript 등 확장자와 무관하게 동작한다."""
    model = torch.jit.load(weights_path, map_location=device)
    model.eval()
    print(f"TorchScript 모델 로드: {weights_path} (device={device})")
    return model


def run(args):
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = load_model(args.weights, device)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {args.source}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (w0, h0))

    prev_time = time.time()
    frame_idx = 0

    # FPS 측정: 처음 --warmup-frames장은 모델/디바이스 워밍업(캐시 미스, CUDA 컨텍스트
    # 초기화 등)으로 느리게 나오므로 평균 계산에서 제외한다. 워밍업이 끝나는 시점부터
    # 시간을 다시 재기 시작해서 순수 추론 구간만 평균을 낸다.
    avg_start_time = None
    avg_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        if args.model_type == "grid":
            tensor = grid_preprocess(frame, args.img_width, args.img_height, device, args.normalize)
        else:
            tensor, scale, pad_x, pad_y = preprocess(frame, args.imgsz, device)

        with torch.no_grad():
            output = model(tensor)

        if args.model_type == "grid":
            # 단일 스케일이면 (1,5,H,W) 텐서 하나, FPN이면 레벨별 텐서 tuple ->
            # 배치 차원만 벗겨서 decode_levels에 전달
            levels = output if isinstance(output, (list, tuple)) else (output,)
            boxes = decode_levels([lv[0] for lv in levels], args.img_width, args.img_height, args.conf)
        else:
            boxes = decode_yolo_output(output, args.conf)
        boxes = nms(boxes, args.nms_iou)

        for xmin, ymin, xmax, ymax, conf in boxes:
            if args.model_type == "grid":
                # 단순 resize(letterbox 아님) 좌표계 -> 원본 프레임 좌표계
                sx, sy = w / args.img_width, h / args.img_height
                rx0 = max(0, min(w, int(xmin * sx)))
                ry0 = max(0, min(h, int(ymin * sy)))
                rx1 = max(0, min(w, int(xmax * sx)))
                ry1 = max(0, min(h, int(ymax * sy)))
            else:
                # letterbox 좌표계 -> 원본 프레임 좌표계
                rx0 = max(0, min(w, int((xmin - pad_x) / scale)))
                ry0 = max(0, min(h, int((ymin - pad_y) / scale)))
                rx1 = max(0, min(w, int((xmax - pad_x) / scale)))
                ry1 = max(0, min(h, int((ymax - pad_y) / scale)))

            cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 255, 0), 2)
            cv2.putText(frame, f"person {conf:.2f}", (rx0, max(0, ry0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        is_warmup = frame_idx < args.warmup_frames
        if not is_warmup:
            if avg_start_time is None:
                avg_start_time = now  # 워밍업 직후 첫 프레임 시점부터 평균 집계 시작
            else:
                avg_frame_count += 1
        avg_fps = avg_frame_count / max(now - avg_start_time, 1e-6) if avg_start_time else 0.0

        cv2.putText(frame, f"FPS: {fps:.1f} (avg {avg_fps:.1f})  persons: {len(boxes)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if writer is not None:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.print_every and frame_idx % args.print_every == 0:
            tag = " [워밍업]" if is_warmup else ""
            print(f"frame {frame_idx}: FPS={fps:.1f}  avg_FPS={avg_fps:.1f}  "
                  f"persons={len(boxes)}{tag}")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    if avg_frame_count > 0:
        total_elapsed = time.time() - avg_start_time
        print(f"\n[FPS 요약] 워밍업 {args.warmup_frames}프레임 제외, "
              f"측정 {avg_frame_count}프레임, 총 {total_elapsed:.2f}초 "
              f"-> 평균 {avg_frame_count / total_elapsed:.2f} FPS")
    else:
        print("\n[FPS 요약] 워밍업 이후 프레임이 없어 평균을 계산할 수 없습니다 "
              f"(영상이 --warmup-frames={args.warmup_frames}보다 짧음).")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 보행자 탐지 실시간 추론")
    parser.add_argument("--source", default="0", help="웹캠 인덱스(예: 0) 또는 영상/이미지 파일 경로")
    parser.add_argument("--weights", default="./models/best.torchscript")
    parser.add_argument("--model-type", choices=["yolo", "grid"], default="yolo",
                        help="yolo=letterbox+이미 디코드된 (1,5,N) 출력 (best.torchscript 등). "
                             "grid=hi/01_custom_model 커스텀 grid 모델(9단계 FPN, 11단계 lite 등, "
                             "raw grid 출력을 sigmoid/exp로 직접 디코드). 모델 종류에 맞게 반드시 지정할 것")
    parser.add_argument("--imgsz", type=int, default=IMGSZ_DEFAULT,
                        help="[yolo 전용] 모델 학습/export 시 사용한 입력 크기와 반드시 일치시킬 것 (320 또는 416)")
    parser.add_argument("--img-width", type=int, default=GRID_IMG_WIDTH_DEFAULT,
                        help="[grid 전용] 모델 학습 시 사용한 입력 너비")
    parser.add_argument("--img-height", type=int, default=GRID_IMG_HEIGHT_DEFAULT,
                        help="[grid 전용] 모델 학습 시 사용한 입력 높이")
    parser.add_argument("--normalize", action="store_true",
                        help="[grid 전용] mobilenet 백본으로 학습한 모델이면 반드시 켤 것 "
                             "(예: 9단계 FPN). custom 백본 모델(11단계 등)은 끄면 됨(기본값)")
    parser.add_argument("--conf", type=float, default=None,
                        help="미지정 시 yolo=0.3, grid=0.05 (학습/평가 때 쓴 기본값)")
    parser.add_argument("--nms-iou", type=float, default=None,
                        help="미지정 시 yolo=0.45, grid=0.3 (학습/평가 때 쓴 기본값)")
    parser.add_argument("--device", default=None, help="미지정 시 cuda 사용 가능하면 cuda, 아니면 cpu")
    parser.add_argument("--save", default=None, help="결과 영상을 저장할 경로 (mp4)")
    parser.add_argument("--no-display", action="store_true", help="GUI 없이 실행 (RPi 헤드리스 환경용)")
    parser.add_argument("--print-every", type=int, default=30,
                        help="콘솔에 FPS를 출력할 프레임 간격 (0이면 출력 안 함). "
                             "--no-display 여부와 무관하게 항상 출력됨")
    parser.add_argument("--warmup-frames", type=int, default=10,
                        help="평균 FPS 계산에서 제외할 초반 프레임 수 (모델/디바이스 초기화 "
                             "오버헤드로 처음 몇 프레임은 느리게 나오므로 평균이 왜곡되는 것을 방지)")
    args = parser.parse_args()

    if args.conf is None:
        args.conf = 0.05 if args.model_type == "grid" else 0.3
    if args.nms_iou is None:
        args.nms_iou = 0.3 if args.model_type == "grid" else 0.45

    run(args)


if __name__ == "__main__":
    main()

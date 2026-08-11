"""
학습된 YOLOv8 보행자 탐지 모델(TorchScript)로 웹캠/영상을 추론하는 스크립트.

RPi4 배포를 염두에 두고 torch, cv2, numpy만 있으면 동작함
(pandas/albumentations 등 학습용 의존성 불필요 — requirements.txt 참고).

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
"""
import argparse
import time

import cv2
import numpy as np
import torch

IMGSZ_DEFAULT = 320
PAD_COLOR = 114  # YOLOv8 표준 letterbox 패딩 색


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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        tensor, scale, pad_x, pad_y = preprocess(frame, args.imgsz, device)

        with torch.no_grad():
            output = model(tensor)

        boxes = decode_yolo_output(output, args.conf)
        boxes = nms(boxes, args.nms_iou)

        for xmin, ymin, xmax, ymax, conf in boxes:
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
        cv2.putText(frame, f"FPS: {fps:.1f}  persons: {len(boxes)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if writer is not None:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif args.print_every and frame_idx % args.print_every == 0:
            print(f"frame {frame_idx}: FPS={fps:.1f} persons={len(boxes)}")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 보행자 탐지 실시간 추론")
    parser.add_argument("--source", default="0", help="웹캠 인덱스(예: 0) 또는 영상/이미지 파일 경로")
    parser.add_argument("--weights", default="./models/best.torchscript")
    parser.add_argument("--imgsz", type=int, default=IMGSZ_DEFAULT,
                        help="모델 학습/export 시 사용한 입력 크기와 반드시 일치시킬 것 (320 또는 416)")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--device", default=None, help="미지정 시 cuda 사용 가능하면 cuda, 아니면 cpu")
    parser.add_argument("--save", default=None, help="결과 영상을 저장할 경로 (mp4)")
    parser.add_argument("--no-display", action="store_true", help="GUI 없이 실행 (RPi 헤드리스 환경용)")
    parser.add_argument("--print-every", type=int, default=30, help="--no-display일 때 콘솔 로그 출력 간격(프레임)")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()

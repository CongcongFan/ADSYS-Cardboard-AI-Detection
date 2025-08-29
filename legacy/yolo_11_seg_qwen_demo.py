#!/usr/bin/env python3
"""
Simple, working baseline:
- Source options: image / video / webcam / DroidCam (phone over Wi‑Fi)
- YOLO11-seg inference @ imgsz=640, IoU=0.20 (reduce duplicate boxes)
- Shows: masks, boxes, class + confidence, FPS and YOLO ms (top‑right)
- Qwen2.5VL runs ASYNC (non‑streaming). We send the *annotated* frame.
- Decision panel overlays after detections; panel background flashes on update
- Prints/overlays Qwen stats when available

Usage examples
--------------
# Image
python yolo_qwen_simple.py --source image --path ./test_img/IMG_5497.JPG

# Video file
python yolo_qwen_simple.py --source video --path ./test_vid/demo.mp4

# Webcam (default index 0)
python yolo_qwen_simple.py --source webcam --cam-index 0

# DroidCam (your phone on Wi‑Fi, default URL http://192.168.0.66:4747/video)
python yolo_qwen_simple.py --source droidcam --droidcam-ip 192.168.0.66 --droidcam-port 4747

Notes
-----
- Keep it simple first; we can add toggles/UI later.
- Requires: ultralytics, opencv-python, requests, numpy, pillow (optional)
- Qwen endpoint defaults to local Ollama (http://localhost:11434/api/generate)
- Close window with 'q'. In image mode, press any key to exit after display.
"""

import argparse
import base64
import json
import time
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO


# --------------------------- Qwen async worker --------------------------- #
class QwenWorker(threading.Thread):
    def __init__(self, url: str, model: str, prompt: str, timeout_s: float = 60.0):
        super().__init__(daemon=True)
        self.url = url
        self.model = model
        self.prompt = prompt
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._latest_job_b64: Optional[str] = None
        self.result = {
            "text": "(Qwen waiting...)",
            "stats": {},
            "qwen_ms": None,
            "updated_at": 0.0,
        }

    def submit_frame(self, bgr_img: np.ndarray):
        # JPEG encode -> base64
        ok, buf = cv2.imencode('.jpg', bgr_img)
        if not ok:
            return
        img_b64 = base64.b64encode(buf).decode('utf-8')
        with self._lock:
            # Keep only the most recent job to avoid backlog
            self._latest_job_b64 = img_b64

    def run(self):
        headers = {"Content-Type": "application/json"}
        while True:
            job_b64 = None
            with self._lock:
                if self._latest_job_b64 is not None:
                    job_b64 = self._latest_job_b64
                    self._latest_job_b64 = None
            if job_b64 is None:
                time.sleep(0.01)
                continue

            payload = {
                "model": self.model,
                "prompt": self.prompt,
                "stream": False,
                "images": [job_b64],
            }
            start = time.perf_counter()
            text = ""
            stats = {}
            try:
                resp = requests.post(self.url, headers=headers, data=json.dumps(payload), timeout=self.timeout_s)
                rj = resp.json()
                text = rj.get("response", "(No response field)")
                # Common Ollama stats (if present)
                stats_keys = [
                    "total_duration", "load_duration",
                    "eval_count", "eval_duration",
                    "prompt_eval_count", "prompt_eval_duration",
                ]
                stats = {k: rj.get(k) for k in stats_keys if k in rj}
            except Exception as e:
                text = f"Qwen error: {e}"
                stats = {}
            q_ms = int((time.perf_counter() - start) * 1000)

            with self._lock:
                self.result = {
                    "text": text,
                    "stats": stats,
                    "qwen_ms": q_ms,
                    "updated_at": time.time(),
                }


# --------------------------- Drawing helpers --------------------------- #

def put_text_with_bg(img: np.ndarray, text: str, org: Tuple[int, int], font_scale=0.6,
                      color=(255, 255, 255), bg=(0, 0, 0), alpha=0.6, thickness=1):
    """Draw text with translucent background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    # Background rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - h - baseline - 4), (x + w + 6, y + 4), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Text
    cv2.putText(img, text, (x + 3, y - 3), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_fps_block(img: np.ndarray, fps: float, yolo_ms: Optional[int], org_top_right=(20, 20)):
    text = f"FPS: {fps:4.1f}"
    if yolo_ms is not None:
        text += f"  |  YOLO: {yolo_ms} ms"
    # Position relative to top-right
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    H, W = img.shape[:2]
    x = W - org_top_right[0] - tw - 10
    y = org_top_right[1] + th + 10
    put_text_with_bg(img, text, (x, y), font_scale=0.6, bg=(32, 32, 32), alpha=0.6)


def draw_boxes_labels(img: np.ndarray, result, primary_idx: int = 0):
    names = result.names if hasattr(result, 'names') else {}
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return

    for i in range(len(boxes)):
        b = boxes.xyxy[i].detach().cpu().numpy().astype(int)
        x1, y1, x2, y2 = b.tolist()
        conf = float(boxes.conf[i].detach().cpu().numpy()) if boxes.conf is not None else 0.0
        cls_id = int(boxes.cls[i].detach().cpu().numpy()) if boxes.cls is not None else -1
        cls_name = names.get(cls_id, str(cls_id))
        label = f"{cls_name} {conf:.2f}"

        # Green for the primary box, cyan for others
        color = (0, 255, 0) if i == primary_idx else (255, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Label above box
        put_text_with_bg(img, label, (x1, y1), font_scale=0.6, bg=(0, 0, 0), alpha=0.6)


def draw_qwen_panel(img: np.ndarray, qwen_res: dict, pos="bl"):
    """Draw Qwen decision block. pos: 'bl' bottom-left or 'tl' top-left."""
    text = qwen_res.get("text", "") or "(empty)"
    q_ms = qwen_res.get("qwen_ms")
    stats = qwen_res.get("stats", {})

    # Compose multi-line message
    lines = ["Qwen decision (JSON):", text]
    if q_ms is not None:
        lines.append(f"Qwen time: {q_ms} ms")
    if stats:
        # show a few key stats if present
        brief = []
        if "eval_count" in stats: brief.append(f"eval_cnt={stats['eval_count']}")
        if "eval_duration" in stats: brief.append(f"eval_ms={int(stats['eval_duration']/1e6)}")
        if "prompt_eval_count" in stats: brief.append(f"prompt_cnt={stats['prompt_eval_count']}")
        if "prompt_eval_duration" in stats: brief.append(f"prompt_ms={int(stats['prompt_eval_duration']/1e6)}")
        if brief:
            lines.append("Stats: " + ", ".join(brief))

    # Flash greenish background for 1.2s after update
    highlight = (time.time() - qwen_res.get("updated_at", 0.0)) < 1.2
    bg = (40, 160, 40) if highlight else (32, 32, 32)

    # Draw as a block with wrapping (simple wrap at ~60 chars)
    wrapped = []
    for L in lines:
        if len(L) <= 60:
            wrapped.append(L)
        else:
            # naive wrap
            for i in range(0, len(L), 60):
                wrapped.append(L[i:i+60])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6
    th = 1
    # Measure block size
    maxw = 0
    totalh = 0
    for L in wrapped:
        (w, h), _ = cv2.getTextSize(L, font, fs, th)
        maxw = max(maxw, w)
        totalh += h + 6
    totalh += 6
    pad = 8
    H, W = img.shape[:2]
    if pos == "bl":
        x0, y0 = 10, H - totalh - 10
    else:  # "tl"
        x0, y0 = 10, 10 + totalh

    # Background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0 - totalh - pad), (x0 + maxw + 2 * pad, y0 + pad), bg, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # Text lines
    y = y0 - totalh + pad + 18
    for L in wrapped:
        cv2.putText(img, L, (x0 + pad, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        y += 18


# --------------------------- Capture helpers --------------------------- #

def open_capture(args) -> Tuple[cv2.VideoCapture, bool]:
    """Return (cap, is_single_frame)."""
    src = args.source.lower()
    if src == 'image':
        img = cv2.imread(args.path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {args.path}")
        return (img, True)
    elif src == 'video':
        cap = cv2.VideoCapture(args.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {args.path}")
        return (cap, False)
    elif src == 'webcam':
        cap = cv2.VideoCapture(int(args.cam_index))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {args.cam_index}")
        return (cap, False)
    elif src == 'droidcam':
        url = f"http://{args.droidcam_ip}:{args.droidcam_port}/video"
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open DroidCam at {url}")
        return (cap, False)
    else:
        raise ValueError("--source must be one of: image, video, webcam, droidcam")


# --------------------------- Main --------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='yolo11l-seg.pt', help='YOLO model path or name')
    ap.add_argument('--source', type=str, required=True, choices=['image', 'video', 'webcam', 'droidcam'])
    ap.add_argument('--path', type=str, default='', help='Image/Video path when source is image/video')
    ap.add_argument('--cam-index', type=int, default=0, help='Webcam index for source=webcam')
    ap.add_argument('--droidcam-ip', type=str, default='192.168.0.66')
    ap.add_argument('--droidcam-port', type=int, default=4747)

    ap.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    ap.add_argument('--iou', type=float, default=0.20, help='YOLO IoU threshold (0.20 to avoid dup boxes)')
    ap.add_argument('--imgsz', type=int, default=640, help='YOLO input size (requirement: 640)')

    ap.add_argument('--qwen-url', type=str, default='http://localhost:11434/api/generate')
    ap.add_argument('--qwen-model', type=str, default='qwen2.5vl:3b')
    ap.add_argument('--prompt', type=str, default=(
        "Describe the cardboard in GREEN bounding box, return only JSON with keys:\n"
        "{\"Warp\": true|false, \"OverallQuality\": one of [good, medium, bad]}\n"
        "Be concise."
    ))

    args = ap.parse_args()

    # Load YOLO model
    model = YOLO(args.model)

    # Open capture
    cap_or_img, single = open_capture(args)

    # Start Qwen worker
    qwen = QwenWorker(args.qwen_url, args.qwen_model, args.prompt)
    qwen.start()

    # FPS tracking
    last_t = time.perf_counter()
    fps = 0.0

    def process_and_show(frame_bgr: np.ndarray) -> bool:
        nonlocal last_t, fps
        t0 = time.perf_counter()
        # Inference (model handles letterbox @ imgsz)
        res = model(frame_bgr, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        yolo_ms = int((time.perf_counter() - t0) * 1000)

        # Mask+box overlay via Ultralytics
        vis = res.plot()  # BGR with masks and boxes

        # Draw our own boxes + labels on top (class + confidence)
        primary_idx = 0
        if getattr(res, 'boxes', None) is not None and len(res.boxes) > 0:
            # choose highest confidence as primary
            try:
                primary_idx = int(np.argmax(res.boxes.conf.detach().cpu().numpy()))
            except Exception:
                primary_idx = 0
            draw_boxes_labels(vis, res, primary_idx=primary_idx)

        # Update FPS
        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if fps > 0 else (1.0 / dt)

        # Stats block (top-right)
        draw_fps_block(vis, fps, yolo_ms)

        # Send annotated frame to Qwen asynchronously (non-blocking). Overwrite old jobs.
        qwen.submit_frame(vis)

        # Draw Qwen decision panel (bottom-left), after all detections so it sits on top
        qres = qwen.result.copy()
        draw_qwen_panel(vis, qres, pos="bl")

        cv2.imshow('YOLO11-seg + Qwen2.5VL (simple)', vis)
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q')

    if single:
        # Single image
        img = cap_or_img
        _quit = process_and_show(img)
        # Wait for a moment to allow Qwen to update once
        print("Press any key to close image window… (or 'q')")
        cv2.waitKey(0)
    else:
        cap: cv2.VideoCapture = cap_or_img
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if process_and_show(frame):
                break
        cap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

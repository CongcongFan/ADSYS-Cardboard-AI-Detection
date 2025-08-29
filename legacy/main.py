# main.py  —  Cardboard stack warp QC (maintainable rewrite)
# Requires: python -m pip install opencv-python numpy matplotlib

from __future__ import annotations
import argparse, json, os, sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import cv2

# ---- Matplotlib (only for interactive clicks) ----
def _setup_matplotlib_backend():
    # Use TkAgg if available; otherwise fall back to Agg (no GUI)
    import matplotlib
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    # ASCII title/text to avoid glyph warnings; user can switch font if needed
    plt.rcParams["axes.unicode_minus"] = False
    return matplotlib, plt

# ----------------------------- Config -----------------------------
IMG_PATH = 'IMG_5490.JPG'  # default image path, can be overridden by CLI

DEFAULT_L_MM = 1080.0    # physical length along x (mm)
DEFAULT_H_MM = 70.0      # physical height along y (mm)

CANNY1, CANNY2 = 50, 150
EDGE_BLUR = 1
MIN_VALID_COL_RATIO = 0.5

# ------------------------- Utils / Helpers ------------------------
def json_safe(val):
    """Convert numpy types to plain Python for JSON."""
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, (tuple,)):
        return [json_safe(v) for v in val]
    if isinstance(val, (dict,)):
        return {k: json_safe(v) for k, v in val.items()}
    return val

def smooth_series(y: np.ndarray, win: int = 31) -> np.ndarray:
    """NaN-aware moving average, odd window."""
    y = y.astype(np.float32, copy=True)
    win = int(win) | 1
    mask = ~np.isnan(y)
    if mask.sum() < 3 or win < 3:
        return y
    idx = np.arange(len(y))
    filled = y.copy()
    filled[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    kernel = np.ones(win, np.float32) / win
    sm = np.convolve(filled, kernel, mode="same")
    sm[~mask] = np.nan
    return sm

def robust_line_fit(x: np.ndarray, y: np.ndarray,
                    max_iter: int = 200, inlier_thresh: float = 2.5):
    """RANSAC-like robust line fit y=a*x+b (returns a,b, inlier_mask)."""
    mask = ~np.isnan(y)
    x0, y0 = x[mask], y[mask]
    if x0.size < 2:
        return np.nan, np.nan, np.zeros_like(y, dtype=bool)

    rng = np.random.default_rng(42)
    best_inliers = None
    best_model = (np.nan, np.nan)
    for _ in range(max_iter):
        if x0.size < 2:
            break
        idx = rng.choice(x0.size, 2, replace=False)
        x_s, y_s = x0[idx], y0[idx]
        if abs(x_s[1] - x_s[0]) < 1e-6:
            continue
        a = (y_s[1] - y_s[0]) / (x_s[1] - x_s[0])
        b = y_s[0] - a * x_s[0]
        resid = np.abs(y0 - (a * x0 + b))
        inliers = resid < inlier_thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_model = (a, b)

    if best_inliers is None or best_inliers.sum() < 2:
        a, b = np.polyfit(x0, y0, 1)
        inlier_mask = mask
        return a, b, inlier_mask

    a, b = np.polyfit(x0[best_inliers], y0[best_inliers], 1)
    inlier_mask = np.zeros_like(y, dtype=bool)
    inlier_mask[np.where(mask)[0][best_inliers]] = True
    return a, b, inlier_mask

def compute_edge_stats(y_edge: np.ndarray, a: float, b: float,
                       px_per_mm: float, length_mm: float):
    xs = np.arange(len(y_edge), dtype=np.float32)
    valid = ~np.isnan(y_edge)
    if not np.any(valid):
        return dict(max_deflection_mm=np.nan, warp_percent=np.nan,
                    mean_abs_resid_mm=np.nan, valid_columns=0)
    y_fit = a * xs + b
    resid_px = np.abs(y_edge[valid] - y_fit[valid])
    max_defl_mm = float(np.nanmax(resid_px) / px_per_mm)
    mean_abs_resid_mm = float(np.nanmean(resid_px) / px_per_mm)
    warp_percent = float((max_defl_mm / length_mm) * 100.0)
    return dict(max_deflection_mm=max_defl_mm,
                warp_percent=warp_percent,
                mean_abs_resid_mm=mean_abs_resid_mm,
                valid_columns=int(valid.sum()))

# --------------------- Perspective & Clicks -----------------------
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners in [lt, rt, rb, lb] order."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    lt = pts[np.argmin(s)]
    rb = pts[np.argmax(s)]
    rt = pts[np.argmin(d)]
    lb = pts[np.argmax(d)]
    return np.array([lt, rt, rb, lb], dtype=np.float32)

def perspective_rectify(img_bgr, corners, px_per_mm, L_mm, H_mm):
    W = int(round(L_mm * px_per_mm))
    H = int(round(H_mm * px_per_mm))
    src = order_corners(corners)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(img_bgr, M, (W, H))
    return rectified, M, (W, H)

def ginput_four_corners(img_rgb):
    _, plt = _setup_matplotlib_backend()
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb); plt.title("Click 4 corners: LT -> RT -> RB -> LB; Enter to confirm")
    pts = plt.ginput(4, timeout=0)
    plt.close()
    if len(pts) != 4:
        raise RuntimeError("Need 4 corner points")
    return np.array(pts, dtype=np.float32)

def calibrate_px_per_mm(img_rgb, known_length_mm):
    _, plt = _setup_matplotlib_backend()
    plt.figure(figsize=(8,6))
    plt.imshow(img_rgb); plt.title("Click two ends of a straight board; Enter to confirm")
    pts = plt.ginput(2, timeout=0)
    plt.close()
    if len(pts) != 2:
        raise RuntimeError("Need 2 points for calibration")
    p1, p2 = np.array(pts[0], np.float32), np.array(pts[1], np.float32)
    px_dist = float(np.linalg.norm(p1 - p2))
    px_per_mm = px_dist / float(known_length_mm)
    print(f"[CAL] px_dist={px_dist:.2f}px, known={known_length_mm:.2f}mm -> {px_per_mm:.5f} px/mm")
    return px_per_mm, (p1, p2)

# -------------------- Edge tracking (DP/Viterbi) -----------------
def _track_edge_band(gray: np.ndarray, band_px=40, which="top",
                     smooth=0.08, max_step=5, bias=0.05, gamma=1.6):
    """Dynamic-programming tracking of a single continuous horizontal edge."""
    H, W = gray.shape
    band_px = int(max(5, min(band_px, H)))

    Gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    Gy = np.abs(Gy)

    if which == "top":
        band = Gy[:band_px, :]
        offset = 0
        pref = np.linspace(0.0, 1.0, band_px, dtype=np.float32)[:, None]
    else:  # bottom
        band = Gy[H-band_px:H, :]
        offset = H - band_px
        pref = np.linspace(0.0, 1.0, band_px, dtype=np.float32)[:, None]

    band_n = band.copy()
    for x in range(W):
        col = band[:, x]
        cmin, cmax = float(col.min()), float(col.max())
        band_n[:, x] = (col - cmin) / (cmax - cmin + 1e-6)

    data_cost = -np.power(band_n, gamma) + bias * pref

    dp = np.empty_like(data_cost)
    prev = np.full_like(data_cost, -1, dtype=np.int32)
    dp[:, 0] = data_cost[:, 0]

    for x in range(1, W):
        for r in range(band_px):
            r0 = max(0, r - max_step)
            r1 = min(band_px - 1, r + max_step)
            chunk = dp[r0:r1+1, x-1]
            diffs = np.arange(r0, r1+1) - r
            score = chunk + smooth * (diffs * diffs)  # quadratic smooth
            idx = int(np.argmin(score))
            best_prev = r0 + idx
            dp[r, x] = data_cost[r, x] + score[idx]
            prev[r, x] = best_prev

    y = np.full(W, np.nan, np.float32)
    r_end = int(np.argmin(dp[:, -1]))
    y[W-1] = offset + r_end
    for x in range(W-1, 0, -1):
        r_end = prev[r_end, x]
        if r_end < 0: break
        y[x-1] = offset + r_end
    return y

def find_edges_viterbi(rectified_gray: np.ndarray, band_top=40, band_bot=40,
                       smooth=0.08, max_step=5, bias=0.05):
    y_top = _track_edge_band(rectified_gray, band_top, "top", smooth, max_step, bias)
    y_bot = _track_edge_band(rectified_gray, band_bot, "bottom", smooth, max_step, bias)
    return y_top, y_bot

def find_edge_columns_band(rectified_gray: np.ndarray, band_top=10, band_bot=10):
    """Simple fallback: pick first edge in top band, last in bottom band."""
    g = rectified_gray.copy()
    if EDGE_BLUR > 0:
        g = cv2.GaussianBlur(g, (2*EDGE_BLUR+1, 2*EDGE_BLUR+1), 0)
    edges = cv2.Canny(g, CANNY1, CANNY2)

    H, W = edges.shape
    y_top = np.full(W, np.nan, np.float32)
    y_bot = np.full(W, np.nan, np.float32)

    top_band = edges[0:min(band_top,H), :]
    bot_band = edges[max(0,H-band_bot):H, :]

    for x in range(W):
        ys_t = np.flatnonzero(top_band[:, x])
        if ys_t.size > 0: y_top[x] = ys_t[0]
        ys_b = np.flatnonzero(bot_band[:, x])
        if ys_b.size > 0: y_bot[x] = (H - band_bot) + ys_b[-1]
    return y_top, y_bot

# ----------------------- Midline & Chord-sag ----------------------
def midline_metrics(y_top, y_bot, px_per_mm, length_mm, smooth_win=31):
    y_mid = (y_top + y_bot) / 2.0
    y_mid = smooth_series(y_mid, win=smooth_win)
    xs = np.arange(len(y_mid), dtype=np.float32)
    mask = ~np.isnan(y_mid)
    if mask.sum() < 2:
        return dict(max_deflection_mm=np.nan, warp_percent=np.nan,
                    mean_abs_resid_mm=np.nan, valid_columns=0,
                    residual_mm_series=None, y_mid=y_mid, fit=(np.nan, np.nan))
    a, b, _ = robust_line_fit(xs, y_mid)
    y_fit = a * xs + b
    resid_px = np.abs(y_mid[mask] - y_fit[mask])
    max_defl_mm = float(np.nanmax(resid_px) / px_per_mm)
    mean_abs_resid_mm = float(np.nanmean(resid_px) / px_per_mm)
    warp_percent = float((max_defl_mm / length_mm) * 100.0)
    resid_full = np.full_like(y_mid, np.nan, np.float32)
    resid_full[mask] = (np.abs(y_mid[mask] - y_fit[mask])) / px_per_mm
    return dict(max_deflection_mm=max_defl_mm,
                warp_percent=warp_percent,
                mean_abs_resid_mm=mean_abs_resid_mm,
                valid_columns=int(mask.sum()),
                residual_mm_series=resid_full,
                y_mid=y_mid, fit=(a, b))

def top_sagitta_from_chord(y_top, px_per_mm, length_mm,
                           margin_pct=0.06, smooth_win=21,
                           use_quantile=True, q=0.3):
    y = smooth_series(y_top, win=smooth_win)
    W = len(y); xs = np.arange(W, dtype=np.float32)
    k = max(5, int(W * margin_pct))
    L0, L1 = k, W - k - 1
    if L1 <= L0: 
        return dict(max_sag_mm=np.nan, sag_percent=np.nan,
                    sag_series_mm=None, valid_columns=0)
    left  = y[L0-min(10,k): L0+min(10,k)]
    right = y[L1-min(10,k): L1+min(10,k)]
    if use_quantile:
        y0 = float(np.nanquantile(left, q))
        y1 = float(np.nanquantile(right, q))
    else:
        y0 = float(np.nanmedian(left)); y1 = float(np.nanmedian(right))
    x0, x1 = float(L0), float(L1)
    slope = (y1 - y0) / (x1 - x0 + 1e-6)
    chord = y0 + slope * (xs - x0)
    sag_px = chord - y           # upward arch only
    sag_px[sag_px < 0] = 0
    mask = ~np.isnan(sag_px); mask[:L0] = False; mask[L1+1:] = False
    if not np.any(mask):
        return dict(max_sag_mm=np.nan, sag_percent=np.nan,
                    sag_series_mm=None, valid_columns=0)
    max_sag_mm = float(np.nanmax(sag_px[mask]) / px_per_mm)
    sag_percent = float((max_sag_mm / length_mm) * 100.0)
    sag_series_mm = np.full(W, np.nan, np.float32)
    sag_series_mm[mask] = sag_px[mask] / px_per_mm
    return dict(max_sag_mm=max_sag_mm, sag_percent=sag_percent,
                sag_series_mm=sag_series_mm, valid_columns=int(mask.sum()))

def proportion_over_threshold_mm(series_mm, thr_mm):
    m = (series_mm is not None) and (~np.isnan(series_mm))
    if isinstance(m, np.ndarray) and m.sum() > 0:
        return float((series_mm[m] >= thr_mm).sum() / m.sum())
    return 0.0

# ------------------------------- Draw -----------------------------
def draw_overlay(rectified_bgr, y_top, y_bot, fit_top, fit_bot,
                 px_per_mm, save_path, extras: dict | None = None):
    vis = rectified_bgr.copy()
    H, W, _ = vis.shape
    xs = np.arange(W, dtype=np.int32)

    # edge points
    for x in xs:
        if not np.isnan(y_top[x]): cv2.circle(vis, (x, int(y_top[x])), 1, (0,255,0), -1)   # green
        if not np.isnan(y_bot[x]): cv2.circle(vis, (x, int(y_bot[x])), 1, (255,0,0), -1)   # blue

    # fits
    a_t, b_t = fit_top
    a_b, b_b = fit_bot
    if not np.isnan(a_t):
        y_t = (a_t*xs + b_t).astype(np.int32)
        for x in xs:
            if 0 <= y_t[x] < H: vis[y_t[x], x] = (0,255,255)   # yellow
    if not np.isnan(a_b):
        y_b = (a_b*xs + b_b).astype(np.int32)
        for x in xs:
            if 0 <= y_b[x] < H: vis[y_b[x], x] = (0,165,255)   # orange

    cv2.putText(vis, f"{px_per_mm:.2f} px/mm", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,40), 2, cv2.LINE_AA)

    if extras:
        y0 = 40
        for k, v in extras.items():
            cv2.putText(vis, f"{k}: {v}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40,40,40), 2, cv2.LINE_AA)
            y0 += 22

    cv2.imwrite(save_path, vis)

# ------------------------------ CLI ------------------------------
@dataclass
class Args:
    image: str
    length_mm: float
    height_mm: float
    threshold: float
    metric: str
    reject_ratio: float
    local_window_mm: float
    local_ratio: float
    band_top_px: int
    band_bot_px: int
    track_smooth: float
    track_max_step: int
    track_bias: float
    px_per_mm: float | None
    calibrate: bool
    known_length_mm: float
    corners: str | None
    use_hsv_envelope: bool
    hsv_lo: str
    hsv_hi: str
    env_guard_px: int


def parse_args() -> Args:
    ap = argparse.ArgumentParser("Cardboard warp QC")
    ap.add_argument("--image", default=IMG_PATH)
    ap.add_argument("--length_mm", type=float, default=DEFAULT_L_MM)
    ap.add_argument("--height_mm", type=float, default=DEFAULT_H_MM)
    ap.add_argument("--threshold", type=float, default=2.0, help="Reject threshold (%)")
    ap.add_argument("--metric", choices=["edge-max", "midline"], default="midline")
    ap.add_argument("--reject_ratio", type=float, default=0.15,
                    help="Midline/top-sag proportion rule (0~1)")
    ap.add_argument("--local_window_mm", type=float, default=300.0)
    ap.add_argument("--local_ratio", type=float, default=0.5)
    ap.add_argument("--band_top_px", type=int, default=40)
    ap.add_argument("--band_bot_px", type=int, default=40)
    ap.add_argument("--track_smooth", type=float, default=0.08)
    ap.add_argument("--track_max_step", type=int, default=5)
    ap.add_argument("--track_bias", type=float, default=0.05)
    ap.add_argument("--px_per_mm", type=float, default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--calibrate", dest="calibrate", action="store_true", help="2-point px/mm calibration")
    g.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    ap.set_defaults(calibrate=True)
    ap.add_argument("--known_length_mm", type=float, default=DEFAULT_L_MM)
    ap.add_argument("--corners", type=str, default=None,
                    help='Optional corners "x1,y1;x2,y2;x3,y3;x4,y4" in LT,RT,RB,LB order')
    ap.add_argument("--use_hsv_envelope", dest="use_hsv_envelope",
                action="store_true", help="Fuse DP edges with HSV envelope (recommended)")
    ap.add_argument("--no-hsv-envelop", dest="use_hsv_envelope",
                    action="store_false")
    ap.set_defaults(use_hsv_envelope=True)

    ap.add_argument("--hsv_lo", type=str, default="5,40,40",
                    help="HSV lower bound for cardboard, e.g., '5,40,40'")
    ap.add_argument("--hsv_hi", type=str, default="30,255,255",
                    help="HSV upper bound for cardboard, e.g., '30,255,255'")
    ap.add_argument("--env_guard_px", type=int, default=6,
                    help="Max pixel jump to envelope when fusing")

    ns = ap.parse_args()
    return Args(**vars(ns))

def cardboard_mask_hsv(bgr, lo=(5,40,40), hi=(30,255,255)):
    """
    简单的纸板(黄/棕)HSV区间掩膜，带一次闭运算去小孔。
    lo/hi 是 (H,S,V)，0-179/0-255/0-255
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(lo, dtype=np.uint8)
    hi = np.array(hi, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def top_bottom_from_mask(mask, band_top_px=40, band_bot_px=40):
    """
    在上/下窄带里，用掩膜的最上/最下像素作为“外轮廓”候选。
    返回 y_top_env, y_bot_env（float32, NaN 表示该列找不到）
    """
    H, W = mask.shape
    y_top = np.full(W, np.nan, np.float32)
    y_bot = np.full(W, np.nan, np.float32)

    # 顶部带
    t1 = 0; t2 = min(H, band_top_px)
    top_band = mask[t1:t2, :]
    for x in range(W):
        ys = np.flatnonzero(top_band[:, x])
        if ys.size:
            y_top[x] = t1 + ys[0]

    # 底部带
    b1 = max(0, H - band_bot_px); b2 = H
    bot_band = mask[b1:b2, :]
    for x in range(W):
        ys = np.flatnonzero(bot_band[:, x])
        if ys.size:
            y_bot[x] = b1 + ys[-1]

    # 轻微平滑
    y_top = smooth_series(y_top, win=11)
    y_bot = smooth_series(y_bot, win=11)
    return y_top, y_bot

def fuse_edges(y_top_dp, y_bot_dp, y_top_env, y_bot_env, guard_px=6):
    """
    将 DP(梯度跟踪) 与 颜色外轮廓 进行融合：
    - 顶边取“更靠上”的那个（y 更小），但与 DP 差距过大时用 guard 限制避免跳到背景噪点；
    - 底边取“更靠下”的那个（y 更大），同理做保护。
    """
    W = max(len(y_top_dp), len(y_top_env))
    def _fuse_top(a, b):
        y = np.copy(a)
        for i in range(W):
            va, vb = a[i], b[i]
            if np.isnan(va) and not np.isnan(vb):
                y[i] = vb
            elif not np.isnan(va) and not np.isnan(vb):
                # 取更靠上的，但最多只向上跳 guard_px
                y[i] = max(va - guard_px, vb) if vb < va else va
            # else: 留 NaN
        return smooth_series(y, win=11)

    def _fuse_bot(a, b):
        y = np.copy(a)
        for i in range(W):
            va, vb = a[i], b[i]
            if np.isnan(va) and not np.isnan(vb):
                y[i] = vb
            elif not np.isnan(va) and not np.isnan(vb):
                # 取更靠下的，但最多只向下跳 guard_px
                y[i] = min(va + guard_px, vb) if vb > va else va
        return smooth_series(y, win=11)

    y_top = _fuse_top(y_top_dp, y_top_env)
    y_bot = _fuse_bot(y_bot_dp, y_bot_env)
    return y_top, y_bot

# ------------------------------ Main -----------------------------
def main():
    args = parse_args()
    img_bgr = cv2.imread(args.image)
    if img_bgr is None: 
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    base = os.path.splitext(os.path.basename(args.image))[0]
    overlay_path = f"{base}_overlay.png"
    report_path  = f"{base}_report.txt"
    json_path    = f"{base}_metrics.json"

    # --- calibration ---
    if args.px_per_mm is None and args.calibrate:
        print("calibrating...")
        args.px_per_mm, cal_pts = calibrate_px_per_mm(img_rgb, args.known_length_mm)
        # optional snapshot of calibration points
        calib_vis = img_bgr.copy()
        for p, color in zip(cal_pts, [(0,0,255), (0,255,255)]):
            cv2.circle(calib_vis, (int(p[0]), int(p[1])), 6, color, -1)
        cv2.imwrite(f"{base}_calib_points.png", calib_vis)
    elif args.px_per_mm is None:
        raise ValueError("px_per_mm is None; use --calibrate or pass --px_per_mm")

    # --- ROI corners ---
    if args.corners:
        pts = []
        for t in args.corners.split(";"):
            x,y = t.split(","); pts.append([float(x), float(y)])
        corners = np.array(pts, dtype=np.float32)
    else:
        corners = ginput_four_corners(img_rgb)

    # --- rectify to physical scale ---
    rectified_bgr, M, (W, H) = perspective_rectify(img_bgr, corners,
                                                   args.px_per_mm,
                                                   args.length_mm,
                                                   args.height_mm)
    gray = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
    xs = np.arange(W, dtype=np.float32)

    # --- edge tracking (with fallback) ---
    y_top, y_bot = find_edges_viterbi(gray,
                                      band_top=args.band_top_px,
                                      band_bot=args.band_bot_px,
                                      smooth=args.track_smooth,
                                      max_step=args.track_max_step,
                                      bias=args.track_bias)
    if args.use_hsv_envelope:
        # 解析阈值
        lo = tuple(int(v) for v in args.hsv_lo.split(","))
        hi = tuple(int(v) for v in args.hsv_hi.split(","))
        mask = cardboard_mask_hsv(rectified_bgr, lo, hi)
        y_top_env, y_bot_env = top_bottom_from_mask(
            mask,
            band_top_px=args.band_top_px,
            band_bot_px=args.band_bot_px
        )
        y_top, y_bot = fuse_edges(y_top, y_bot, y_top_env, y_bot_env,
                                guard_px=args.env_guard_px)
    
    def _edge_std(y): 
        m = ~np.isnan(y); 
        return float(np.nanstd(y[m])) if m.sum()>0 else 0.0
    std_bot = _edge_std(y_bot)
    if std_bot < 0.6:   # auto-relax & fallback
        y_top2, y_bot2 = find_edges_viterbi(gray,
                                            band_top=max(args.band_top_px, 50),
                                            band_bot=max(args.band_bot_px, 60),
                                            smooth=max(0.02, args.track_smooth*0.5),
                                            max_step=max(args.track_max_step, 6),
                                            bias=max(0.02, args.track_bias*0.6))
        if _edge_std(y_bot2) > std_bot:
            y_top, y_bot = y_top2, y_bot2
        else:
            y_top, y_bot = find_edge_columns_band(gray,
                                                  band_top=max(args.band_top_px, 40),
                                                  band_bot=max(args.band_bot_px, 40))
            y_top = smooth_series(y_top, 21); y_bot = smooth_series(y_bot, 21)

    # --- fits for top/bot (always available for audit) ---
    a_top, b_top, _ = robust_line_fit(xs, y_top)
    a_bot, b_bot, _ = robust_line_fit(xs, y_bot)
    stat_top = compute_edge_stats(y_top, a_top, b_top, args.px_per_mm, args.length_mm)
    stat_bot = compute_edge_stats(y_bot, a_bot, b_bot, args.px_per_mm, args.length_mm)

    valid_ratio = ((~np.isnan(y_top)).sum() + (~np.isnan(y_bot)).sum()) / (2.0 * W)


    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"Time: {ts}",
        f"Image: {args.image}",
        f"Rectified size: {W}x{H} px | Scale: {args.px_per_mm:.3f} px/mm",
        f"Physical: {args.length_mm:.1f} mm (L) x {args.height_mm:.1f} mm (H)",
        f"Valid ratio: {valid_ratio:.3f}",
        f"Threshold: {args.threshold:.2f} %"
    ]

    thr_mm = args.threshold * args.length_mm / 100.0

    if valid_ratio < MIN_VALID_COL_RATIO:
        status = "INVALID_MEASUREMENT"
        extras = dict()
        draw_overlay(rectified_bgr, y_top, y_bot, (a_top,b_top), (a_bot,b_bot),
                     args.px_per_mm, overlay_path, extras)
        lines.append("Decision: INVALID_MEASUREMENT (too few valid edge points)")
        metrics = dict(time=ts, image=args.image, decision=status,
                       scale_px_per_mm=float(args.px_per_mm),
                       length_mm=float(args.length_mm),
                       height_mm=float(args.height_mm),
                       valid_ratio=float(valid_ratio),
                       threshold_percent=float(args.threshold))
    else:
        # Midline
        mid = midline_metrics(y_top, y_bot, args.px_per_mm, args.length_mm, smooth_win=31)
        ratio_mid = proportion_over_threshold_mm(mid["residual_mm_series"], thr_mm)

        # Top chord sagitta
        top_sag = top_sagitta_from_chord(y_top, args.px_per_mm, args.length_mm,
                                         margin_pct=0.06, smooth_win=21, use_quantile=True, q=0.3)
        ratio_top_sag = proportion_over_threshold_mm(top_sag["sag_series_mm"], thr_mm)

        # Local window (midline residual)
        win_px = int(args.local_window_mm * args.px_per_mm)
        local_trigger = False
        resid = mid["residual_mm_series"]
        if resid is not None and win_px > 2:
            for start in range(0, len(resid) - win_px + 1, max(1, win_px // 4)):
                seg = resid[start:start+win_px]
                m = ~np.isnan(seg)
                if m.sum() == 0: continue
                over = (seg[m] >= thr_mm).sum() / m.sum()
                if over >= args.local_ratio:
                    local_trigger = True
                    break

        # Optional strict edge-max (for audit)
        overall_max_edges = max(stat_top["max_deflection_mm"], stat_bot["max_deflection_mm"])
        overall_warp_edges = (overall_max_edges / args.length_mm) * 100.0

        # Decision
        reject_top_fit = stat_top["max_deflection_mm"] >= thr_mm
        reject_top_sag = (top_sag["max_sag_mm"] >= thr_mm) or (ratio_top_sag >= args.reject_ratio)
        reject_mid = (mid["max_deflection_mm"] >= thr_mm) and (ratio_mid >= args.reject_ratio)
        reject_edge_max = (args.metric == "edge-max") and (overall_warp_edges >= args.threshold)

        status = "REJECT" if (reject_top_fit or reject_top_sag or reject_mid or local_trigger or reject_edge_max) else "PASS"

        # Overlay text
        extras = {
            "mid_max(mm)": f"{mid['max_deflection_mm']:.1f}",
            "mid_warp(%)": f"{mid['warp_percent']:.2f}",
            f"mid_over>={thr_mm:.1f}mm": f"{ratio_mid*100:.1f}%",
            "topSag_max(mm)": f"{top_sag['max_sag_mm']:.1f}",
            "topSag(%)": f"{top_sag['sag_percent']:.2f}",
        }
        draw_overlay(rectified_bgr, y_top, y_bot, (a_top,b_top), (a_bot,b_bot),
                     args.px_per_mm, overlay_path, extras)

        lines += [
            f"Top: max_deflection={stat_top['max_deflection_mm']:.2f} mm, warp={stat_top['warp_percent']:.3f} %, "
            f"mean_abs_resid={stat_top['mean_abs_resid_mm']:.2f} mm, cols={stat_top['valid_columns']}",
            f"Bot: max_deflection={stat_bot['max_deflection_mm']:.2f} mm, warp={stat_bot['warp_percent']:.3f} %, "
            f"mean_abs_resid={stat_bot['mean_abs_resid_mm']:.2f} mm, cols={stat_bot['valid_columns']}",
            f"Midline: max_deflection={mid['max_deflection_mm']:.2f} mm, warp={mid['warp_percent']:.3f} %, "
            f"mean_abs_resid={mid['mean_abs_resid_mm']:.2f} mm, cols={mid['valid_columns']}",
            f"Mid over >= {thr_mm:.1f} mm: {ratio_mid*100:.1f} %",
            f"TopSag (chord): max={top_sag['max_sag_mm']:.2f} mm ({top_sag['sag_percent']:.2f} %), "
            f"over >= {thr_mm:.1f} mm: {ratio_top_sag*100:.1f} %",
            f"Local window: {args.local_window_mm:.0f}mm, ratio>={args.local_ratio:.2f} -> {local_trigger}",
            f"Edge-max overall: {overall_warp_edges:.3f} %",
            f"Decision: {status}"
        ]

        # JSON metrics (summaries only; no big arrays)
        metrics = {
            "time": ts,
            "image": args.image,
            "scale_px_per_mm": float(args.px_per_mm),
            "length_mm": float(args.length_mm),
            "height_mm": float(args.height_mm),
            "valid_ratio": float(valid_ratio),
            "threshold_percent": float(args.threshold),
            "threshold_mm": float(thr_mm),
            "decision": status,
            "top": json_safe(stat_top),
            "bottom": json_safe(stat_bot),
            "midline": {
                "max_deflection_mm": float(mid["max_deflection_mm"]),
                "warp_percent": float(mid["warp_percent"]),
                "mean_abs_resid_mm": float(mid["mean_abs_resid_mm"]),
                "valid_columns": int(mid["valid_columns"]),
                "proportion_over_threshold": float(ratio_mid),
            },
            "top_sag": {
                "max_sag_mm": float(top_sag["max_sag_mm"]),
                "sag_percent": float(top_sag["sag_percent"]),
                "proportion_over_threshold": float(ratio_top_sag),
            },
            "local_window": {
                "window_mm": float(args.local_window_mm),
                "ratio_threshold": float(args.local_ratio),
                "triggered": bool(local_trigger),
            },
            "edge_max_overall_percent": float(overall_warp_edges),
        }

    # --- write outputs ---
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(metrics), f, indent=2, ensure_ascii=False)

    print(f"[OK] overlay: {overlay_path}")
    print(f"[OK] report : {report_path}")
    print(f"[OK] metrics: {json_path}")
    print(f"[RESULT] {metrics['decision']}")

if __name__ == "__main__":
    main()

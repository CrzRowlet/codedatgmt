"""
src/utils/visualizer.py
Tiện ích vẽ dùng chung toàn hệ thống.
"""
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional


# ── Màu sắc theo class ────────────────────────────────────
CLASS_COLORS_BGR = {
    "car"       : (50,  200, 50),
    "motorbike" : (0,   165, 255),
    "person"    : (255, 80,  0),
    "truck"     : (200, 0,   220),
}
DEFAULT_COLOR = (150, 150, 150)

ALERT_COLORS_BGR = {
    "SAFE"    : (50,  200, 50),
    "WARNING" : (0,   165, 255),
    "DANGER"  : (0,   0,   230),
}
ALERT_BG_BGR = {
    "SAFE"    : (15,  60,  15),
    "WARNING" : (15,  50, 100),
    "DANGER"  : (40,   0,   0),
}
ALERT_TEXT_VI = {
    "SAFE"    : "AN TOAN",
    "WARNING" : "CANH BAO",
    "DANGER"  : "NGUY HIEM !!!",
}


def put_text_bg(
    frame : np.ndarray,
    text  : str,
    pos   : Tuple[int, int],
    color : Tuple[int, int, int] = (255, 255, 255),
    bg    : Tuple[int, int, int] = (20, 20, 20),
    scale : float = 0.55,
    thick : int   = 1,
    alpha : float = 0.65,
) -> None:
    """Vẽ text với nền mờ lên frame (in-place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    # Nền
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 2, y - th - 4), (x + tw + 4, y + baseline + 2), bg, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # Text
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def draw_bbox(
    frame     : np.ndarray,
    bbox      : Tuple[int, int, int, int],
    label     : str,
    color     : Tuple[int, int, int] = (100, 200, 100),
    thickness : int   = 2,
    corner_len: int   = 12,
) -> None:
    """
    Vẽ bounding box với góc nổi bật (corner style).
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Góc
    cl = min(corner_len, (x2 - x1) // 4, (y2 - y1) // 4)
    for px, py, dx, dy in [
        (x1, y1,  cl,  0), (x1, y1,  0,  cl),
        (x2, y1, -cl,  0), (x2, y1,  0,  cl),
        (x1, y2,  cl,  0), (x1, y2,  0, -cl),
        (x2, y2, -cl,  0), (x2, y2,  0, -cl),
    ]:
        cv2.line(frame, (px, py), (px + dx, py + dy), color, max(1, thickness))

    # Label
    put_text_bg(frame, label, (x1, max(y1 - 6, 14)), color=color, scale=0.52)


def draw_alert_panel(
    frame   : np.ndarray,
    level   : str,
    dist_m  : float,
    ttc_s   : Optional[float],
    n_objs  : int,
    fps     : float,
    x: int = 8, y: int = 8,
    width: int = 280, height: int = 100,
) -> None:
    """Vẽ panel thông tin chính ở góc trên-trái."""
    color   = ALERT_COLORS_BGR.get(level, (150, 150, 150))
    bg      = ALERT_BG_BGR.get(level, (20, 20, 20))
    label   = ALERT_TEXT_VI.get(level, level)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), bg, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

    cv2.putText(frame, label, (x + 10, y + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Dist: {dist_m:.1f}m",
                (x + 10, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    ttc_txt = f"TTC: {ttc_s:.1f}s" if ttc_s is not None else "TTC: --"
    cv2.putText(frame, ttc_txt,
                (x + 150, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Objs: {n_objs}",
                (x + 10, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 140, 140), 1, cv2.LINE_AA)

    fps_color = (50, 200, 50) if fps >= 15 else (0, 165, 255) if fps >= 8 else (0, 0, 220)
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (x + 210, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.46, fps_color, 1, cv2.LINE_AA)


def draw_danger_border(frame: np.ndarray, flash_hz: float = 2.5) -> None:
    """Vẽ viền đỏ nhấp nháy khi DANGER."""
    if int(time.time() * flash_hz) % 2 == 0:
        h, w = frame.shape[:2]
        for t in (4, 9):
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), t)


def draw_distance_bar(
    frame     : np.ndarray,
    distance_m: float,
    max_dist  : float = 20.0,
    x: int = None, y: int = None, w: int = 12, h: int = 120,
) -> None:
    """Vẽ thanh khoảng cách dọc ở cạnh phải frame."""
    fh, fw = frame.shape[:2]
    if x is None: x = fw - 28
    if y is None: y = fh // 2 - h // 2

    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

    ratio  = max(0.0, min(1.0, 1.0 - distance_m / max_dist))
    bar_h  = int(h * ratio)
    if bar_h > 0:
        if distance_m <= 3.0:
            color = (0, 0, 220)
        elif distance_m <= 8.0:
            color = (0, 165, 255)
        else:
            color = (50, 200, 50)
        cv2.rectangle(frame, (x, y + h - bar_h), (x + w, y + h), color, -1)

    put_text_bg(frame, f"{distance_m:.0f}m", (x - 10, y + h + 14),
                color=(200, 200, 200), scale=0.42)


def draw_footer(frame: np.ndarray, frame_count: int) -> None:
    """Vẽ timestamp + frame count ở góc dưới-phải."""
    h, w = frame.shape[:2]
    text = f"{time.strftime('%H:%M:%S')}  |  frame #{frame_count}"
    cv2.putText(frame, text, (w - 210, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

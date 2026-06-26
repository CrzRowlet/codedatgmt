"""
=============================================================
GIAO DIEN DEMO GUI
src/ui/dashboard.py
=============================================================
Giao dien bao gom:
  - Khung video chinh voi overlay real-time
  - Dong ho TTC va khoang cach theo thoi gian thuc
  - Thanh trang thai mau sac theo muc canh bao
  - Bieu do khoang cach theo thoi gian (matplotlib)
  - Nut tai video, chinh nguong, xem log CSV
  - Panel thong ke (FPS, tong canh bao, session)

Cach dung:
  python src/ui/dashboard.py
  python src/ui/dashboard.py --video "demo/videos/video1.mp4"
"""

import cv2
import sys
import os
import time
import json
import threading
import queue
import argparse
from pathlib import Path
from collections import deque
import numpy as np

# Import cac module CWS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.tracking.ttc       import ObjectTracker, draw_full_overlay, AlertState
from src.alert.alert_system  import AlertSystem, AlertLevel, AlertOverlayRenderer, EventLogger
from src.lane.lane_roi       import LaneROI, LanePosition, draw_lane_overlay

# ── Kiem tra Tkinter ──────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ── Kiem tra matplotlib ───────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")   # Khong can display
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════
#  CONFIG GIAO DIEN
# ══════════════════════════════════════════════════════════
UI_CONFIG = {
    "window_title" : "He Thong Canh Bao Va Cham - CWS Demo",
    "video_w"      : 960,
    "video_h"      : 540,
    "chart_w"      : 960,
    "chart_h"      : 200,
    "history_len"  : 100,   # So diem du lieu giu tren bieu do
    "update_ms"    : 33,    # ~30 FPS update UI
}

COLORS_TK = {
    "bg_dark"    : "#1A1A2E",
    "bg_panel"   : "#16213E",
    "bg_card"    : "#0F3460",
    "safe"       : "#27AE60",
    "warning"    : "#F39C12",
    "danger"     : "#E74C3C",
    "text_main"  : "#ECF0F1",
    "text_muted" : "#95A5A6",
    "accent"     : "#3498DB",
    "border"     : "#2C3E50",
}


# ══════════════════════════════════════════════════════════
#  PROCESSOR THREAD (chay pipeline o thread rieng)
# ══════════════════════════════════════════════════════════
class PipelineProcessor(threading.Thread):
    """
    Chay vong lap xu ly frame o thread rieng.
    Giao tiep voi UI qua queue.
    """

    def __init__(
        self,
        source       : str,
        model_path   : str  = "models/weights/best.pt",
        enable_audio : bool = True,
        result_queue : queue.Queue = None,
    ):
        super().__init__(daemon=True)
        self.source       = source
        self.model_path   = model_path
        self.enable_audio = enable_audio
        self.q            = result_queue or queue.Queue(maxsize=4)
        self._stop_event  = threading.Event()
        self._paused      = threading.Event()
        self.conf_threshold = 0.5
        self.enable_lane  = False
        self.alert        = None

    def stop(self):
        self._stop_event.set()
        if self.alert:
            self.alert.stop()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def run(self):
        # Khoi tao module
        detector  = self._load_yolo()
        tracker   = ObjectTracker(max_missed=8)
        self.alert = AlertSystem() if self.enable_audio else None
        renderer  = AlertOverlayRenderer()
        logger    = EventLogger()
        self.lane_roi = LaneROI(config_file="configs/lane_roi.json", n_lanes=3)

        # Mo video
        try:
            src = int(self.source)
        except ValueError:
            src = self.source

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.q.put({"error": f"Khong mo duoc: {self.source}"})
            return

        frame_count = 0
        fps_hist    = deque(maxlen=30)

        while not self._stop_event.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue

            t0  = time.time()
            ret, frame = cap.read()
            if not ret:
                self.q.put({"eof": True})
                break

            frame_count += 1

            # YOLO detect
            if detector:
                dets = self._yolo_detect(detector, frame)
            else:
                dets = self._demo_dets(frame, frame_count)

            # Track + TTC
            tracks = tracker.update(dets, frame.shape[1], timestamp=t0)
            if self.lane_roi and self.enable_lane:
                self.lane_roi.update_frame_size(frame.shape[1], frame.shape[0])
                tracks = self.lane_roi.classify_tracks(tracks)
                alert_tracks = [t for t in tracks if t.get("lane_position") in (LanePosition.EGO_LANE, LanePosition.UNKNOWN)]
            else:
                alert_tracks = tracks

            # Alert
            level = AlertLevel.SAFE
            if alert_tracks and self.alert:
                level = self.alert.process_tracks(alert_tracks)
                if level != AlertLevel.SAFE and alert_tracks:
                    t = alert_tracks[0]
                    logger.log(level, t["distance_m"], t.get("ttc"), t.get("class_name",""))

            # Render
            if self.lane_roi and self.enable_lane:
                frame = draw_lane_overlay(frame, self.lane_roi, tracks)
            vis = renderer.render(frame, tracks, level,
                                   fps=self._calc_fps(fps_hist), frame_count=frame_count)

            # Ghi FPS
            elapsed = time.time() - t0
            fps_hist.append(elapsed)

            # Dua vao queue (drop neu day)
            result = {
                "frame"       : vis,
                "tracks"      : tracks,
                "level"       : level,
                "fps"         : self._calc_fps(fps_hist),
                "frame_count" : frame_count,
                "alert_stats" : self.alert.get_stats() if self.alert else {},
            }
            try:
                self.q.put_nowait(result)
            except queue.Full:
                pass

        cap.release()

    def _load_yolo(self):
        if not Path(self.model_path).exists():
            return None
        try:
            from ultralytics import YOLO
            return YOLO(self.model_path)
        except Exception:
            return None

    def _yolo_detect(self, model, frame) -> list:
        results = model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
        dets = []
        for box in results.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            dets.append({"bbox":(x1,y1,x2,y2), "class_id":int(box.cls[0]), "conf":float(box.conf[0])})
        return dets

    def _demo_dets(self, frame, frame_count) -> list:
        h, w = frame.shape[:2]
        d = max(0.05, 1.0 - frame_count * 0.012)
        bw = int(w * 0.25 / d)
        bh = int(h * 0.18 / d)
        cx, cy = w//2, h//2+20
        x1,y1 = max(0,cx-bw//2), max(0,cy-bh//2)
        x2,y2 = min(w-1,cx+bw//2), min(h-1,cy+bh//2)
        return [{"bbox":(x1,y1,x2,y2), "class_id":0, "conf":0.88}]

    def _calc_fps(self, hist) -> float:
        if not hist: return 0.0
        return 1.0 / (sum(hist)/len(hist))


# ══════════════════════════════════════════════════════════
#  BIEU DO MATPLOTLIB EMBED
# ══════════════════════════════════════════════════════════
class DistanceChart:
    """Bieu do khoang cach + TTC theo thoi gian, render thanh numpy array."""

    def __init__(self, width=960, height=200, history_len=100):
        self.w          = width
        self.h          = height
        self.max_pts    = history_len
        self.dist_hist  = deque(maxlen=history_len)
        self.ttc_hist   = deque(maxlen=history_len)
        self.time_hist  = deque(maxlen=history_len)
        self.t_start    = time.time()
        self._fig       = None
        self._canvas    = None
        self._init_fig()

    def _init_fig(self):
        if not HAS_MPL:
            return
        plt.style.use("dark_background")
        self._fig, (self._ax1, self._ax2) = plt.subplots(
            1, 2, figsize=(self.w/100, self.h/100), dpi=100,
            gridspec_kw={"width_ratios": [2, 1]}
        )
        self._fig.patch.set_facecolor("#0F3460")
        for ax in (self._ax1, self._ax2):
            ax.set_facecolor("#16213E")
            ax.tick_params(colors="#95A5A6", labelsize=7)
            ax.spines["bottom"].set_color("#2C3E50")
            ax.spines["left"].set_color("#2C3E50")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        self._canvas = FigureCanvasAgg(self._fig)

    def update(self, distance_m: float, ttc_s: float = None):
        t = time.time() - self.t_start
        self.dist_hist.append(distance_m)
        self.ttc_hist.append(ttc_s if ttc_s is not None else float("nan"))
        self.time_hist.append(t)

    def render(self) -> np.ndarray:
        """Tra ve numpy array (BGR) de dung voi OpenCV."""
        if not HAS_MPL or not self.dist_hist:
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)

        times = list(self.time_hist)
        dists = list(self.dist_hist)
        ttcs  = list(self.ttc_hist)

        # Bieu do 1: Khoang cach
        ax1 = self._ax1
        ax1.clear()
        ax1.set_facecolor("#16213E")
        ax1.plot(times, dists, color="#3498DB", linewidth=1.5, label="Distance (m)")
        # Vung canh bao
        ax1.axhline(y=8.0,  color="#F39C12", linewidth=0.8, linestyle="--", alpha=0.7)
        ax1.axhline(y=3.0,  color="#E74C3C", linewidth=0.8, linestyle="--", alpha=0.7)
        ax1.fill_between(times, 0, 3.0, alpha=0.15, color="#E74C3C")
        ax1.fill_between(times, 3.0, 8.0, alpha=0.08, color="#F39C12")
        ax1.set_ylabel("Dist (m)", color="#95A5A6", fontsize=8)
        ax1.set_xlabel("Time (s)", color="#95A5A6", fontsize=8)
        ax1.set_title("Khoang Cach Theo Thoi Gian", color="#ECF0F1", fontsize=9, pad=4)
        ax1.tick_params(colors="#95A5A6", labelsize=7)
        ax1.spines["bottom"].set_color("#2C3E50")
        ax1.spines["left"].set_color("#2C3E50")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        if dists:
            ax1.set_ylim(0, max(max(dists)*1.2, 15))

        # Bieu do 2: TTC
        ax2 = self._ax2
        ax2.clear()
        ax2.set_facecolor("#16213E")
        valid_ttc = [(t, v) for t, v in zip(times, ttcs) if not (isinstance(v, float) and np.isnan(v))]
        if valid_ttc:
            tt, tv = zip(*valid_ttc)
            colors_ttc = ["#E74C3C" if v < 2 else "#F39C12" if v < 5 else "#27AE60" for v in tv]
            ax2.scatter(tt, tv, c=colors_ttc, s=12, alpha=0.8)
            ax2.plot(tt, tv, color="#7F8C8D", linewidth=0.8, alpha=0.5)
        ax2.axhline(y=5.0, color="#27AE60", linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.axhline(y=2.0, color="#E74C3C", linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.set_ylabel("TTC (s)", color="#95A5A6", fontsize=8)
        ax2.set_xlabel("Time (s)", color="#95A5A6", fontsize=8)
        ax2.set_title("TTC Theo Thoi Gian", color="#ECF0F1", fontsize=9, pad=4)
        ax2.tick_params(colors="#95A5A6", labelsize=7)
        ax2.spines["bottom"].set_color("#2C3E50")
        ax2.spines["left"].set_color("#2C3E50")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_ylim(0, 12)

        # Render ra numpy (tuong thich matplotlib >= 3.8)
        self._fig.tight_layout(pad=0.5)
        self._canvas.draw()
        w_px, h_px = self._canvas.get_width_height()
        try:
            # matplotlib >= 3.8
            buf = np.frombuffer(self._canvas.buffer_rgba(), dtype=np.uint8)
            img = buf.reshape(h_px, w_px, 4)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        except AttributeError:
            # fallback matplotlib < 3.8
            buf = np.frombuffer(self._canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore
            img = buf.reshape(h_px, w_px, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (self.w, self.h))
        return img


# ══════════════════════════════════════════════════════════
#  DASHBOARD TKINTER
# ══════════════════════════════════════════════════════════
class CWSDashboard:
    """
    Giao dien Tkinter day du voi:
      - Khung video chinh
      - Panel thong so (dist, TTC, FPS, thong ke)
      - Bieu do khoang cach/TTC embed
      - Thanh trang thai mau sac
      - Nut dieu khien
    """

    def __init__(self, source: str = "0", model_path: str = "models/weights/best.pt",
                 enable_audio: bool = True):
        if not HAS_TK:
            raise ImportError("Can tkinter. Cai dat: pip install tk")

        self.source       = source
        self.model_path   = model_path
        self.enable_audio = enable_audio

        # State
        self._q        = queue.Queue(maxsize=4)
        self._processor = None
        self._running   = False
        self._paused    = False
        self._stats_hist = {"fps": deque(maxlen=60), "warning": 0, "danger": 0}
        self._chart     = DistanceChart(width=UI_CONFIG["chart_w"], height=UI_CONFIG["chart_h"])

        # Tkinter
        self.root = tk.Tk()
        self._build_ui()

    def _build_ui(self):
        self.root.title(UI_CONFIG["window_title"])
        self.root.configure(bg=COLORS_TK["bg_dark"])
        self.root.resizable(True, True)

        # ── Thanh tieu de ────────────────────────────────
        title_bar = tk.Frame(self.root, bg=COLORS_TK["bg_panel"], height=50)
        title_bar.pack(fill="x", side="top")
        tk.Label(title_bar, text="HE THONG CANH BAO VA CHAM  |  CWS v1.0",
                 font=("Arial", 14, "bold"), bg=COLORS_TK["bg_panel"],
                 fg=COLORS_TK["text_main"]).pack(side="left", padx=20, pady=10)
        self._lbl_status = tk.Label(title_bar, text="● DUNG",
                                     font=("Arial", 12, "bold"),
                                     bg=COLORS_TK["bg_panel"], fg=COLORS_TK["text_muted"])
        self._lbl_status.pack(side="right", padx=20)

        # ── Thanh trang thai & Dieu khien (Pack truoc de ghim) ──
        self._status_bar = tk.Frame(self.root, height=8)
        self._status_bar.pack(fill="x", side="bottom")
        self._update_status_bar(AlertLevel.SAFE)

        ctrl = tk.Frame(self.root, bg=COLORS_TK["bg_panel"], height=50)
        ctrl.pack(fill="x", side="bottom")
        self._build_controls(ctrl)

        # ── Noi dung chinh (Pack sau de expand) ──────────────
        main = tk.Frame(self.root, bg=COLORS_TK["bg_dark"])
        main.pack(fill="both", expand=True, padx=10, pady=5)

        # Cot trai: video + chart
        left = tk.Frame(main, bg=COLORS_TK["bg_dark"])
        left.pack(side="left", fill="both", expand=True)

        # Bieu do (Pack truoc o duoi de dam bao luon hien thi)
        self._lbl_chart = tk.Label(left, bg=COLORS_TK["bg_panel"])
        self._lbl_chart.pack(side="bottom", fill="x", pady=(0, 5))

        # Khung video (Container để chặn vòng lặp phóng to)
        self._video_container = tk.Frame(left, bg="black")
        self._video_container.pack(side="top", pady=(0, 5), expand=True, fill="both")
        
        self._blank_video = tk.PhotoImage(width=UI_CONFIG["video_w"], height=UI_CONFIG["video_h"])
        self._lbl_video = tk.Label(self._video_container, bg="black", image=self._blank_video)
        self._lbl_video.place(relwidth=1.0, relheight=1.0)

        # Cot phai: panel thong so
        right = tk.Frame(main, bg=COLORS_TK["bg_panel"], width=240)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)
        self._build_right_panel(right)

    def _build_right_panel(self, parent):
        def card(parent, title, var_name, unit="", color=COLORS_TK["text_main"]):
            f = tk.Frame(parent, bg=COLORS_TK["bg_card"],
                         relief="flat", bd=0, highlightbackground=COLORS_TK["border"],
                         highlightthickness=1)
            f.pack(fill="x", padx=8, pady=4)
            tk.Label(f, text=title, font=("Arial", 9),
                     bg=COLORS_TK["bg_card"], fg=COLORS_TK["text_muted"]).pack(anchor="w", padx=8, pady=(6,0))
            var = tk.StringVar(value="--")
            lbl = tk.Label(f, textvariable=var, font=("Arial", 22, "bold"),
                           bg=COLORS_TK["bg_card"], fg=color)
            lbl.pack(anchor="center", padx=8)
            if unit:
                tk.Label(f, text=unit, font=("Arial", 9),
                         bg=COLORS_TK["bg_card"], fg=COLORS_TK["text_muted"]).pack(anchor="e", padx=8, pady=(0,6))
            else:
                tk.Label(f, text="", bg=COLORS_TK["bg_card"]).pack(pady=(0,4))
            return var

        tk.Label(parent, text="THONG SO HIEN TAI", font=("Arial", 11, "bold"),
                 bg=COLORS_TK["bg_panel"], fg=COLORS_TK["accent"]).pack(pady=(15,5))

        self._var_dist    = card(parent, "Khoang Cach",   "dist",  "met",   COLORS_TK["accent"])
        self._var_ttc     = card(parent, "TTC",           "ttc",   "giay",  COLORS_TK["warning"])
        self._var_alert   = card(parent, "Trang Thai",    "alert", "",      COLORS_TK["safe"])
        self._var_fps     = card(parent, "FPS",           "fps",   "fps",   COLORS_TK["text_main"])
        self._var_objects = card(parent, "Vat The",       "objs",  "objects",COLORS_TK["text_main"])

        # Thong ke
        tk.Label(parent, text="THONG KE PHIEN", font=("Arial", 10, "bold"),
                 bg=COLORS_TK["bg_panel"], fg=COLORS_TK["accent"]).pack(pady=(15,5))
        self._var_total    = card(parent, "Tong Frame",    "total",  "frames", COLORS_TK["text_main"])
        self._var_n_warn   = card(parent, "Canh Bao",      "warn",   "",       COLORS_TK["warning"])
        self._var_n_danger = card(parent, "Nguy Hiem",     "danger", "",       COLORS_TK["danger"])

    def _build_controls(self, parent):
        style_btn = {"font": ("Arial", 10, "bold"), "bd": 0, "padx": 16, "pady": 8,
                     "relief": "flat", "cursor": "hand2"}

        self._btn_start = tk.Button(parent, text="  CHAY  ",
            bg=COLORS_TK["safe"], fg="white",
            command=self._toggle_run, **style_btn)
        self._btn_start.pack(side="left", padx=8, pady=6)

        tk.Button(parent, text="Tai Video",
            bg=COLORS_TK["bg_card"], fg=COLORS_TK["text_main"],
            command=self._load_video, **style_btn).pack(side="left", padx=4, pady=6)

        tk.Button(parent, text="Xem Log",
            bg=COLORS_TK["bg_card"], fg=COLORS_TK["text_main"],
            command=self._open_log, **style_btn).pack(side="left", padx=4, pady=6)

        tk.Button(parent, text="Snapshot",
            bg=COLORS_TK["bg_card"], fg=COLORS_TK["text_main"],
            command=self._snapshot, **style_btn).pack(side="left", padx=4, pady=6)

        # Slider confidence
        tk.Label(parent, text="Confidence:", bg=COLORS_TK["bg_panel"],
                 fg=COLORS_TK["text_muted"], font=("Arial", 9)).pack(side="left", padx=(10,4))
        self._conf_var = tk.DoubleVar(value=0.5)
        tk.Scale(parent, variable=self._conf_var, from_=0.2, to=0.9, resolution=0.05,
                 orient="horizontal", length=100, bg=COLORS_TK["bg_panel"],
                 fg=COLORS_TK["text_main"], highlightthickness=0, bd=0).pack(side="left")

        # Model selection
        tk.Label(parent, text="Model:", bg=COLORS_TK["bg_panel"],
                 fg=COLORS_TK["text_muted"], font=("Arial", 9)).pack(side="left", padx=(15,4))
        self._model_var = tk.StringVar(value="Fine-tune (models/weights/best.pt)")
        model_combo = ttk.Combobox(parent, textvariable=self._model_var, 
                                   values=["Fine-tune (models/weights/best.pt)", "Scratch (scratch/Yolo_from_scratch/best.pt)"],
                                   state="readonly", width=35)
        model_combo.pack(side="left", padx=4)

        # Lane toggle
        self._enable_lane_var = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text="Phân làn", variable=self._enable_lane_var,
                       bg=COLORS_TK["bg_panel"], fg=COLORS_TK["text_main"],
                       selectcolor=COLORS_TK["bg_card"], bd=0).pack(side="left", padx=(10,4))

    # ── Xu ly su kien ────────────────────────────────────
    def _toggle_run(self):
        if not self._running:
            self._start()
        else:
            self._stop()

    def _start(self):
        self._running = True
        self._btn_start.config(text="  DUNG  ", bg=COLORS_TK["danger"])
        self._lbl_status.config(text="● DANG CHAY", fg=COLORS_TK["safe"])

        selected_model = self._model_var.get()
        if "Fine-tune" in selected_model:
            self.model_path = "models/weights/best.pt"
        else:
            self.model_path = "scratch/Yolo_from_scratch/best.pt"

        self._processor = PipelineProcessor(
            source       = self.source,
            model_path   = self.model_path,
            enable_audio = self.enable_audio,
            result_queue = self._q,
        )
        self._processor.start()
        self._poll_queue()

    def _stop(self):
        self._running = False
        if self._processor:
            self._processor.stop()
        self._btn_start.config(text="  CHAY  ", bg=COLORS_TK["safe"])
        self._lbl_status.config(text="● DUNG", fg=COLORS_TK["text_muted"])
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def _load_video(self):
        path = filedialog.askopenfilename(
            title="Chon video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tat ca", "*.*")]
        )
        if path:
            self.source = path
            if self._running:
                self._stop()
                time.sleep(0.3)
                self._start()

    def _open_log(self):
        log_path = "logs/events.csv"
        if os.path.exists(log_path):
            import subprocess, platform
            if platform.system() == "Windows":
                os.startfile(log_path)
            elif platform.system() == "Darwin":
                subprocess.call(["open", log_path])
            else:
                subprocess.call(["xdg-open", log_path])
        else:
            messagebox.showinfo("Thong bao", "Chua co file log. Hay chay pipeline truoc.")

    def _snapshot(self):
        if not hasattr(self, "_last_frame"):
            messagebox.showinfo("Info", "Khong co hinh anh de chup!")
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, self._last_frame)
            messagebox.showinfo("Thanh cong", f"Da luu {path}")

    # ── Poll queue va cap nhat UI ─────────────────────────
    def _poll_queue(self):
        """Lay frame tu queue va cap nhat UI. Goi lai chinh no moi 33ms."""
        if not self._running:
            return
        if self._processor:
            self._processor.conf_threshold = self._conf_var.get()
            self._processor.enable_lane = self._enable_lane_var.get()

        try:
            data = self._q.get_nowait()
            if "error" in data:
                messagebox.showerror("Loi", data["error"])
                self._stop()
                return
            if "eof" in data:
                self._stop()
                return

            self._update_ui(data)
        except queue.Empty:
            pass

        if self._running:
            self.root.after(UI_CONFIG["update_ms"], self._poll_queue)

    def _update_ui(self, data: dict):
        """Cap nhat toan bo UI tu data."""
        frame  = data["frame"]
        tracks = data["tracks"]
        level  = data["level"]
        fps    = data["fps"]
        fcount = data["frame_count"]
        astats = data.get("alert_stats", {})

        # 1. Cap nhat frame video
        self._update_video_frame(frame)

        # 2. Cap nhat bieu do
        if tracks:
            top = tracks[0]
            self._chart.update(top["distance_m"], top.get("ttc"))
        chart_img = self._chart.render()
        self._update_chart(chart_img)

        # 3. Cap nhat panel thong so
        if tracks:
            top = tracks[0]
            self._var_dist.set(f"{top['distance_m']:.1f}")
            self._var_ttc.set(f"{top['ttc']:.1f}" if top.get("ttc") else "--")
            self._var_objects.set(str(len(tracks)))
        else:
            self._var_dist.set("--")
            self._var_ttc.set("--")
            self._var_objects.set("0")

        self._var_alert.set(level.value)
        self._var_fps.set(f"{fps:.0f}")
        self._var_total.set(str(fcount))
        self._var_n_warn.set(str(astats.get("warning", 0)))
        self._var_n_danger.set(str(astats.get("danger", 0)))

        # 4. Mau sac alert
        self._update_status_bar(level)

    def _update_video_frame(self, frame: np.ndarray):
        """Chuyen frame OpenCV BGR -> Tkinter PhotoImage va hien thi."""
        self._last_frame = frame.copy()
        try:
            from PIL import Image, ImageTk
            
            lbl_w = self._video_container.winfo_width()
            lbl_h = self._video_container.winfo_height()
            
            if lbl_w > 10 and lbl_h > 10:
                w, h = lbl_w, lbl_h
            else:
                w, h = UI_CONFIG["video_w"], UI_CONFIG["video_h"]
                
            frame_rgb = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self._lbl_video.configure(image=img)
            self._lbl_video._img = img   # Giu tham chieu
        except ImportError:
            # Fallback: hien thi bang cv2 window rieng
            cv2.imshow("CWS Video", frame)
            cv2.waitKey(1)

    def _update_chart(self, chart_img: np.ndarray):
        """Cap nhat bieu do."""
        try:
            from PIL import Image, ImageTk
            img_rgb = cv2.cvtColor(chart_img, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            self._lbl_chart.configure(image=img)
            self._lbl_chart._img = img
        except ImportError:
            pass

    def _update_status_bar(self, level: AlertLevel):
        color = COLORS_TK["safe"] if level == AlertLevel.SAFE else \
                COLORS_TK["warning"] if level == AlertLevel.WARNING else COLORS_TK["danger"]
        self._status_bar.configure(bg=color)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self._stop()
        self.root.destroy()


# ══════════════════════════════════════════════════════════
#  FALLBACK: Demo chi dung OpenCV (khong can Tkinter)
# ══════════════════════════════════════════════════════════
class CWSOpenCVDemo:
    """
    Demo don gian chi dung OpenCV (khong can Tkinter/PIL).
    Hien thi: video + overlay + bieu do tren cung 1 cua so.
    """

    def __init__(self, source: str, model_path: str = "models/weights/best.pt",
                 enable_audio: bool = True):
        self.source       = source
        self.model_path   = model_path
        self.enable_audio = enable_audio
        self._q           = queue.Queue(maxsize=4)
        self._chart       = DistanceChart(width=UI_CONFIG["video_w"],
                                           height=UI_CONFIG["chart_h"])

    def run(self):
        from src.tracking.ttc      import ObjectTracker, draw_full_overlay
        from src.alert.alert_system import AlertSystem, AlertLevel, AlertOverlayRenderer

        tracker  = ObjectTracker()
        alert    = AlertSystem() if self.enable_audio else None
        renderer = AlertOverlayRenderer()
        logger   = EventLogger()

        detector = self._load_yolo()

        try:
            src = int(self.source)
        except ValueError:
            src = self.source

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"  Khong mo duoc: {self.source}")
            return

        frame_count = 0
        fps_hist    = deque(maxlen=30)

        print("  Phim tat: Q=Thoat | S=Snapshot | SPACE=Pause")

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect
            if detector:
                dets = self._yolo_detect(detector, frame)
            else:
                dets = self._demo_dets(frame, frame_count)

            # Track
            tracks = tracker.update(dets, frame.shape[1], timestamp=t0)

            # Alert
            level = AlertLevel.SAFE
            if tracks and alert:
                level = alert.process_tracks(tracks)
                if level != AlertLevel.SAFE and tracks:
                    t = tracks[0]
                    logger.log(level, t["distance_m"], t.get("ttc"), t.get("class_name",""))

            # FPS
            elapsed = time.time() - t0
            fps_hist.append(elapsed)
            fps = 1.0 / (sum(fps_hist)/len(fps_hist))

            # Render video
            vis = renderer.render(frame, tracks, level,
                                   fps=fps, frame_count=frame_count)

            # Resize video cho vua man hinh
            vw, vh = UI_CONFIG["video_w"], UI_CONFIG["video_h"]
            vis_resized = cv2.resize(vis, (vw, vh))

            # Render bieu do
            if tracks:
                top = tracks[0]
                self._chart.update(top["distance_m"], top.get("ttc"))
            chart = self._chart.render()

            # Ghep video + bieu do
            combined = np.vstack([vis_resized, chart])

            cv2.imshow("CWS Demo - He Thong Canh Bao Va Cham", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                ts = time.strftime("%H%M%S")
                path = f"demo/outputs/snapshot_{ts}.jpg"
                Path("demo/outputs").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(path, combined)
                print(f"  Snapshot: {path}")
            elif key == ord(" "):
                cv2.waitKey(0)

        cap.release()
        cv2.destroyAllWindows()
        if alert:
            print(f"\n  Thong ke: {alert.get_stats()}")

    def _load_yolo(self):
        if not Path(self.model_path).exists():
            print(f"  Chua co model {self.model_path} - dung demo mode")
            return None
        try:
            from ultralytics import YOLO
            return YOLO(self.model_path)
        except Exception:
            return None

    def _yolo_detect(self, model, frame) -> list:
        results = model.predict(frame, conf=0.5, verbose=False)[0]
        return [{"bbox": tuple(map(int, b.xyxy[0])), "class_id": int(b.cls[0]), "conf": float(b.conf[0])}
                for b in results.boxes]

    def _demo_dets(self, frame, frame_count) -> list:
        h, w = frame.shape[:2]
        d    = max(0.05, 1.0 - frame_count * 0.010)
        bw   = int(w * 0.25 / d)
        bh   = int(h * 0.18 / d)
        cx, cy = w//2, h//2+20
        return [{"bbox": (max(0,cx-bw//2), max(0,cy-bh//2),
                          min(w-1,cx+bw//2), min(h-1,cy+bh//2)),
                 "class_id": 0, "conf": 0.88}]


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Giao dien Demo CWS")
    parser.add_argument("--source", "--video", dest="source", default="0",
                        help="Video file hoac camera index (0=webcam)")
    parser.add_argument("--model",     default="models/weights/best.pt")
    parser.add_argument("--no-audio",  action="store_true")
    parser.add_argument("--opencv",    action="store_true",
                        help="Dung che do OpenCV thuan tuy (khong Tkinter)")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════╗
║   GIAO DIEN DEMO CWS                                 ║
╚══════════════════════════════════════════════════════╝""")

    enable_audio = not args.no_audio

    if args.opencv or not HAS_TK:
        print("  Che do: OpenCV Demo (khong Tkinter)\n")
        demo = CWSOpenCVDemo(
            source       = args.source,
            model_path   = args.model,
            enable_audio = enable_audio,
        )
        demo.run()
    else:
        print("  Che do: Tkinter Dashboard\n")
        try:
            app = CWSDashboard(
                source       = args.source,
                model_path   = args.model,
                enable_audio = enable_audio,
            )
            app.run()
        except ImportError as e:
            print(f"  Loi khoi dong Tkinter: {e}")
            print("  Chuyen sang che do OpenCV...\n")
            demo = CWSOpenCVDemo(
                source       = args.source,
                model_path   = args.model,
                enable_audio = enable_audio,
            )
            demo.run()

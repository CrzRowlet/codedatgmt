"""
=============================================================
PIPELINE TICH HOP HOAN CHINH
main.py  (entry point chinh)
=============================================================
Ket hop:
  YOLOv8 detection
  Distance estimation
  Object tracking + TTC
  Alert system + TTS

Cach dung:
  python main.py --source demo\videos\video1.mp4
  python main.py --source 0                         # Webcam
  python main.py --source video.mp4 --save          # Luu video ket qua
  python main.py --source video.mp4 --no-audio      # Tat am thanh
  python main.py --benchmark                        # Do FPS + do chinh xac
"""

import cv2
import time
import sys
import os
import argparse
import threading
import queue
import json
from pathlib import Path
from datetime import datetime

# Đảm bảo in UTF-8 trên Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ── Import cac module da xay dung ────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.distance.estimator  import DistanceEstimator
from src.tracking.ttc        import ObjectTracker, draw_full_overlay, AlertState
from src.alert.alert_system  import AlertSystem, AlertLevel, AlertOverlayRenderer
from src.utils.logger        import EventLogger, FPSCounter, get_logger
from src.utils.visualizer    import draw_danger_border, draw_footer
from src.lane.lane_roi       import LaneROI, LanePosition, draw_lane_overlay

_log = get_logger("Pipeline")

# ── Cau hinh chay ────────────────────────────────────────
PIPELINE_CONFIG = {
    "model_path"     : "models/weights/best.pt",
    "confidence"     : 0.50,
    "iou_threshold"  : 0.45,
    "img_size"       : 640,
    "device"         : "auto",
    "target_fps"     : 20,
    "save_path"      : "demo/outputs/result.mp4",
    "log_path"       : "logs/events.csv",
    "show_window"    : True,
    "print_stats"    : True,
    # ── Lane ROI ─────────────────────────────────────────
    "lane_roi_enabled"  : True,          # Bật/tắt phân làn
    "lane_roi_config"   : "configs/lane_roi.json",
    "lane_n_lanes"      : 3,             # Số làn: 1 / 2 / 3
    "lane_ego_only"     : False,         # True = chỉ cảnh báo xe ở EGO lane
}


# ══════════════════════════════════════════════════════════
#  CLASS CHINH: CollisionWarningPipeline
# ══════════════════════════════════════════════════════════
class CollisionWarningPipeline:
    """
    Pipeline tich hop hoan chinh:
      Video/Camera → YOLO → Distance → TTC → Alert → Overlay → Display

    Su dung thread rieng cho am thanh de khong block inference.
    """

    def __init__(
        self,
        source       : str,
        model_path   : str  = PIPELINE_CONFIG["model_path"],
        confidence   : float = PIPELINE_CONFIG["confidence"],
        save_output  : bool  = False,
        enable_audio : bool  = True,
        show_window  : bool  = True,
        config       : dict  = None,
    ):
        self.source       = source
        self.model_path   = model_path
        self.confidence   = confidence
        self.save_output  = save_output
        self.enable_audio = enable_audio
        self.show_window  = show_window
        self.cfg          = config or PIPELINE_CONFIG.copy()

        # Ket qua runtime
        self._frame_count  = 0
        self._fps_counter  = FPSCounter(window=30)
        self._running      = False
        self._cap          = None
        self._writer       = None

        # Khoi tao cac module
        print("\n[Pipeline] Khoi tao cac module...")
        self._init_modules()

    def _init_modules(self):
        """Khoi tao tuan tu cac module."""
        # 1. YOLO detector
        print("  [1/4] Load YOLOv8 model...")
        self.detector = self._load_yolo()

        # 2. Tracker + distance
        print("  [2/4] Khoi tao ObjectTracker...")
        self.tracker  = ObjectTracker(max_missed=10)

        # 3. Alert system
        print("  [3/4] Khoi tao AlertSystem...")
        self.alert    = AlertSystem() if self.enable_audio else None
        self.renderer = AlertOverlayRenderer()
        self.logger   = EventLogger(self.cfg["log_path"])

        # 4. Ket qua hien tai (thread-safe)
        self._current_tracks : list = []
        self._current_level   = AlertLevel.SAFE
        self._lock            = threading.Lock()

        # 5. Lane ROI — phân làn đường
        print("  [4/5] Khoi tao LaneROI...")
        self._init_lane_roi()

        print("  [5/5] Tat ca module san sang!\n")

    def _load_yolo(self):
        """Load YOLOv8 model. Tra ve None neu chua co."""
        if not Path(self.model_path).exists():
            print(f"  ⚠  Chua co model: {self.model_path}")
            print("     Chay Ngay 4 truoc: python scripts/day4_train.py --all")
            print("     Dung che do demo (khong co YOLO)...\n")
            return None

        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            print(f"  ✅ Model loaded: {self.model_path}")
            return model
        except ImportError:
            print("  pip install ultralytics")
            return None
        except Exception as e:
            print(f"  ✗ Load model loi: {e}")
            return None

    def _init_lane_roi(self):
        """
        Khởi tạo Lane ROI module.
        Load config nếu có, nếu không dùng ROI mặc định.
        Người dùng có thể thiết lập ROI thủ công bằng:
            python main.py --setup-lanes
        """
        if not self.cfg.get("lane_roi_enabled", True):
            self.lane_roi = None
            print("  ⚠  Lane ROI bị tắt (lane_roi_enabled=False)")
            return

        self.lane_roi = LaneROI(
            config_file = self.cfg.get("lane_roi_config", "configs/lane_roi.json"),
            n_lanes     = self.cfg.get("lane_n_lanes", 3),
        )

        if self.lane_roi.lanes:
            print(f"  ✅ LaneROI: {len(self.lane_roi.lanes)} làn sẵn sàng")
        else:
            print("  ⚠  LaneROI: Chưa có ROI — dùng ROI mặc định")
            print("     Để vẽ ROI thực tế: python main.py --setup-lanes")

    # ── Mo nguon video ─────────────────────────────────────
    def _open_source(self):
        try:
            src = int(self.source)
        except ValueError:
            src = self.source

        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Khong mo duoc nguon: {self.source}")

        width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        total  = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  Nguon       : {self.source}")
        print(f"  Do phan giai: {width}x{height}  |  FPS goc: {fps:.1f}")
        if total > 0:
            print(f"  Tong frame  : {total}  ({total/fps:.1f}s)")

        # Cập nhật kích thước frame cho LaneROI (scale ROI nếu cần)
        if self.lane_roi is not None:
            self.lane_roi.update_frame_size(width, height)

        return width, height, fps

    def _init_writer(self, width: int, height: int, fps: float):
        if not self.save_output:
            return
        Path(self.cfg["save_path"]).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.cfg["save_path"], fourcc, fps, (width, height)
        )
        print(f"  Luu video   : {self.cfg['save_path']}")

    # ── Xu ly mot frame ───────────────────────────────────
    def _process_frame(self, frame: 'np.ndarray', timestamp: float) -> 'np.ndarray':
        """Pipeline xu ly mot frame: detect → track → lane_roi → alert → render."""

        # Buoc 1: YOLO Detection
        if self.detector is not None:
            detections = self._yolo_detect(frame)
        else:
            detections = self._demo_detections(frame)

        # Buoc 2: Tracker + Distance + TTC
        tracks = self.tracker.update(
            detections  = detections,
            frame_width = frame.shape[1],
            timestamp   = timestamp,
        )

        # Buoc 3: Phan loai lane cho tung xe
        #   – Xe trong EGO lane → giữ nguyên mức cảnh báo
        #   – Xe ở lane bên    → giảm mức ưu tiên
        if self.lane_roi is not None:
            tracks = self.lane_roi.classify_tracks(tracks)
            # Nếu bật chế độ ego_only: lọc bỏ xe không ở EGO lane
            if self.cfg.get("lane_ego_only", False):
                tracks = [t for t in tracks
                          if t.get("lane_position") in (
                              LanePosition.EGO_LANE, LanePosition.UNKNOWN)]

        # Buoc 4: Kich hoat canh bao
        level = AlertLevel.SAFE
        if tracks and self.alert:
            level = self.alert.process_tracks(tracks)

            # Log su kien WARNING/DANGER
            if level != AlertLevel.SAFE:
                top = tracks[0]
                self.logger.log(
                    alert_level    = level.value,
                    distance_m     = top["distance_m"],
                    ttc_s          = top.get("ttc"),
                    class_name     = top.get("class_name",""),
                    track_id       = top.get("track_id", -1),
                    approach_speed = top.get("approach_speed", 0.0),
                )

        # Luu trang thai (thread-safe)
        with self._lock:
            self._current_tracks = tracks
            self._current_level  = level

        # Buoc 5: Render overlay
        tracks_for_render = [
            {**t, "alert_state": t["alert_state"]}
            for t in tracks
        ]
        vis = self.renderer.render(
            frame         = frame,
            tracks_output = tracks_for_render,
            alert_level   = level,
            fps           = self._current_fps,
            frame_count   = self._frame_count,
        )

        # Buoc 6: Ve Lane ROI len frame (sau khi ve bbox)
        if self.lane_roi is not None:
            vis = draw_lane_overlay(vis, self.lane_roi, tracks)

        # Buoc 7: Ve lane stats nho goc tren-phai
        if self.lane_roi is not None and tracks:
            self._draw_lane_stats(vis, tracks)

        return vis

    def _yolo_detect(self, frame) -> list:
        """Chay YOLO inference va tra ve danh sach detection."""
        device = self.cfg["device"]
        if device == "auto":
            device = None  # De YOLO tu dong lua chon thiet bi toi uu (GPU neu co, con lai CPU)

        results = self.detector.predict(
            source  = frame,
            conf    = self.confidence,
            iou     = self.cfg["iou_threshold"],
            imgsz   = self.cfg["img_size"],
            device  = device,
            verbose = False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox"    : (x1, y1, x2, y2),
                "class_id": int(box.cls[0]),
                "conf"    : float(box.conf[0]),
            })
        return detections

    def _demo_detections(self, frame) -> list:
        """
        Tao detection gia khi chua co model.
        Mô phong mot xe dang lai den gan.
        """
        h, w = frame.shape[:2]
        t    = self._frame_count / 30.0
        # Xe di chuyen tu xa den gan (simulate)
        import math
        dist_factor = max(0.05, 1.0 - t * 0.015)
        bw = int(w * 0.3 * (1.0 / dist_factor))
        bh = int(h * 0.2 * (1.0 / dist_factor))
        cx, cy = w//2, h//2 + 30
        x1 = max(0, cx - bw//2)
        y1 = max(0, cy - bh//2)
        x2 = min(w-1, cx + bw//2)
        y2 = min(h-1, cy + bh//2)
        return [{"bbox": (x1, y1, x2, y2), "class_id": 0, "conf": 0.88}]

    def _draw_lane_stats(self, frame: 'np.ndarray', tracks: list):
        """
        Vẽ bảng thống kê số xe theo làn ở góc trên-phải.
        Hiển thị: tên làn + số xe + màu tương ứng.
        """
        if not self.lane_roi or not self.lane_roi.lanes:
            return

        stats  = self.lane_roi.get_lane_stats(tracks)
        h, w   = frame.shape[:2]
        panel_w = 170
        panel_h = 18 + len(self.lane_roi.lanes) * 20 + 6
        px = w - panel_w - 8
        py = 8

        # Nền panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (60, 60, 60), 1)

        cv2.putText(frame, "LANE STATS", (px + 8, py + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        for i, lane in enumerate(self.lane_roi.lanes):
            n   = stats.get(lane.name, 0)
            txt = f"{lane.name}: {n} xe"
            color = lane.color if n == 0 else (0, 100, 255) if lane.is_ego else (0, 200, 255)
            y_txt = py + 30 + i * 20
            cv2.putText(frame, txt, (px + 8, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    @property
    def _current_fps(self) -> float:
        return self._fps_counter.fps

    # ══════════════════════════════════════════════════════
    #  VONG LAP CHINH
    # ══════════════════════════════════════════════════════
    def run(self):
        """Chay pipeline chinh."""
        width, height, src_fps = self._open_source()
        self._init_writer(width, height, src_fps)

        self._running = True
        frame_delay   = 1.0 / self.cfg["target_fps"]

        print("\n" + "="*50)
        print("  PIPELINE DANG CHAY  -  Nhan Q de thoat")
        print("="*50 + "\n")

        t_start = time.time()

        try:
            while self._running:
                t_frame = time.time()

                ret, frame = self._cap.read()
                if not ret:
                    print("\n  Het video / Mat ket noi camera.")
                    break

                self._frame_count += 1

                # Xu ly frame
                vis = self._process_frame(frame, t_frame)

                # Ghi video output
                if self._writer:
                    self._writer.write(vis)

                # Hien thi
                if self.show_window:
                    cv2.imshow("He Thong Canh Bao Va Cham - CWS v1.0", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    elif key == ord("s"):   # S: chup anh
                        self._snapshot(vis)
                    elif key == ord("p"):   # P: pause
                        cv2.waitKey(0)

                # Tinh FPS
                elapsed = time.time() - t_frame
                self._fps_counter.tick()

                # In trang thai dinh ky (moi 2 giay)
                if self.cfg["print_stats"] and self._frame_count % int(src_fps*2) == 0:
                    self._print_status()

        except KeyboardInterrupt:
            print("\n  Dung boi nguoi dung (Ctrl+C)")
        finally:
            self._cleanup(t_start)

    def _snapshot(self, frame):
        """Luu anh chup man hinh."""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"demo/outputs/snapshot_{ts}.jpg"
        Path("demo/outputs").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, frame)
        print(f"  Snapshot: {path}")

    def _print_status(self):
        """In trang thai hien tai ra terminal."""
        fps   = self._current_fps
        level = self._current_level.value
        with self._lock:
            n_objs = len(self._current_tracks)
            if self._current_tracks:
                top = self._current_tracks[0]
                detail = f"{top['class_name']} {top['distance_m']:.1f}m"
                if top.get("ttc"):
                    detail += f" TTC={top['ttc']:.1f}s"
            else:
                detail = "Khong co vat can"

        icon = {"SAFE": "✅", "WARNING": "⚡", "DANGER": "🚨"}.get(level, "")
        print(f"  Frame {self._frame_count:5d} | FPS {fps:5.1f} | {icon} {level:8s} | {n_objs} objects | {detail}")

    def _cleanup(self, t_start):
        """Dong tat ca resource."""
        elapsed = time.time() - t_start
        avg_fps = self._frame_count / elapsed if elapsed > 0 else 0

        if self._cap:
            self._cap.release()
        if self._writer:
            self._writer.release()
        if self.show_window:
            cv2.destroyAllWindows()

        # Luu thong ke
        self.logger.save_session_summary(
            extra={
                "total_frames": self._frame_count,
                "duration_s"  : round(elapsed, 1),
                "avg_fps"     : round(avg_fps, 1),
            }
        )

        print(f"\n{'='*50}")
        print(f"  PHIEN LAM VIEC KET THUC")
        print(f"  Tong frame   : {self._frame_count}")
        print(f"  Thoi gian    : {elapsed:.1f}s")
        print(f"  FPS trung binh: {avg_fps:.1f}")
        if self.alert:
            s = self.alert.get_stats()
            print(f"  Canh bao     : {s.get('warning',0)} WARNING, {s.get('danger',0)} DANGER")
        if self.save_output:
            print(f"  Video luu tai: {self.cfg['save_path']}")
        print(f"  Log luu tai  : {self.cfg['log_path']}")
        print(f"{'='*50}\n")


# ══════════════════════════════════════════════════════════
#  BENCHMARK
# ══════════════════════════════════════════════════════════
def benchmark(model_path: str, video_path: str, n_frames: int = 100):
    """Do FPS va kiem tra pipeline end-to-end."""
    print(f"\n  BENCHMARK - {n_frames} frames")
    print(f"  Model : {model_path}")
    print(f"  Video : {video_path}\n")

    pipeline = CollisionWarningPipeline(
        source       = video_path,
        model_path   = model_path,
        enable_audio = False,
        show_window  = False,
        config       = {**PIPELINE_CONFIG, "print_stats": False},
    )

    cap = cv2.VideoCapture(video_path if video_path != "0" else 0)
    if not cap.isOpened():
        print("  Khong mo duoc video!")
        return

    times    = []
    t_detect = []
    t_track  = []

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret: break

        t0 = time.time()
        # Detect
        t1 = time.time()
        dets = pipeline._yolo_detect(frame) if pipeline.detector else []
        t2 = time.time()
        # Track
        tracks = pipeline.tracker.update(dets, frame.shape[1], timestamp=t0)
        t3 = time.time()

        times.append(t3 - t0)
        t_detect.append(t2 - t1)
        t_track.append(t3 - t2)

    cap.release()

    if not times: return

    import numpy as np
    avg_total  = np.mean(times) * 1000
    avg_detect = np.mean(t_detect) * 1000
    avg_track  = np.mean(t_track) * 1000
    fps        = 1000 / avg_total

    print(f"  {'Buoc':<20} {'TB (ms)':>10} {'P95 (ms)':>10}")
    print(f"  {'─'*42}")
    print(f"  {'YOLO Detect':<20} {avg_detect:>10.1f} {np.percentile(t_detect,95)*1000:>10.1f}")
    print(f"  {'Track + TTC':<20} {avg_track:>10.1f} {np.percentile(t_track,95)*1000:>10.1f}")
    print(f"  {'TONG':<20} {avg_total:>10.1f} {np.percentile(times,95)*1000:>10.1f}")
    print(f"  {'─'*42}")
    print(f"  FPS uoc tinh: {fps:.1f} {'✅' if fps >= 15 else '⚠ (< 15 FPS, can toi uu)'}")

    if fps < 15:
        print("\n  Goi y toi uu FPS:")
        print("    1. Giam img_size: 416 thay vi 640")
        print("    2. Dung yolov8n thay vi yolov8s")
        print("    3. Bat half=True (FP16) trong predict()")
        print("    4. Giam resolution video dau vao xuong 720p")
        print("    5. Skip frame: xu ly 1/2 frame")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="He thong canh bao va cham - Main Pipeline"
    )
    parser.add_argument("--source",    default="ask",
        help="Nguon video: path file, index (0=webcam), hoac ask de hoi tuong tac")
    parser.add_argument("--model",     default=None,
        help="Duong dan model YOLO (.pt). Neu de trong se hoi tuong tac.")
    parser.add_argument("--model-type", choices=["tune", "scratch", "ask"], default="ask",
        help="Chon loai model de chay nhanh ma khong can hoi")
    parser.add_argument("--conf",      type=float, default=PIPELINE_CONFIG["confidence"],
        help="Nguong confidence YOLO (mac dinh: 0.5)")
    parser.add_argument("--save",      action="store_true",
        help="Luu video ket qua ra demo/outputs/result.mp4")
    parser.add_argument("--no-audio",  action="store_true",
        help="Tat canh bao am thanh")
    parser.add_argument("--no-window", action="store_true",
        help="Tat hien thi cua so (dung khi chay server)")
    parser.add_argument("--benchmark", action="store_true",
        help="Do FPS va hieu suat pipeline")
    parser.add_argument("--fps",       type=int, default=PIPELINE_CONFIG["target_fps"],
        help="FPS muc tieu (mac dinh: 20)")
    # ── Lane ROI ──────────────────────────────────────────
    parser.add_argument("--setup-lanes", action="store_true",
        help="Vẽ ROI làn đường tương tác bằng chuột rồi thoát")
    parser.add_argument("--lanes",       type=int, default=3,
        help="Số làn đường: 1 / 2 / 3 (mặc định: 3)")
    parser.add_argument("--ego-only",    action="store_true",
        help="Chỉ cảnh báo xe nằm trong EGO lane")
    return parser.parse_args()


def main():
    args = parse_args()

    print("""
╔══════════════════════════════════════════════════════╗
║   HE THONG CANH BAO VA CHAM - CWS v1.0              ║
║   Collision Warning System                          ║
║   YOLO + Distance + TTC + Alert                     ║
╚══════════════════════════════════════════════════════╝""")

    # ── Chọn Nguồn Hình Ảnh (Source) ────────────────────────
    source = args.source
    if source == "ask":
        print("\n  [?] BẠN MUỐN LẤY HÌNH ẢNH TỪ NGUỒN NÀO?")
        print("      1. Webcam của máy tính (Mặc định)")
        print("      2. Camera điện thoại (thông qua IP Webcam)")
        print("      3. File video có sẵn")
        try:
            choice_src = input("  Nhập lựa chọn (1/2/3) [Enter để chọn 1]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  [Tự động] Bỏ qua chọn, sử dụng Webcam.")
            choice_src = "1"
        
        if choice_src == "2":
            ip = input("  >> Nhập địa chỉ IP trên điện thoại (vd: 192.168.1.15): ").strip()
            # Tự động làm sạch nếu người dùng copy dư chữ http:// hoặc :8080
            ip = ip.replace("http://", "").replace("/video", "").split(":")[0].strip()
            if not ip: ip = "192.168.1.15"
            source = f"http://{ip}:8080/video"
        elif choice_src == "3":
            video_path = input("  >> Nhập đường dẫn file video [Enter để chạy file demo mặc định]: ").strip()
            source = video_path if video_path else "demo/videos/video1.mp4"
        else:
            source = "0"

    # ── Chọn Model ─────────────────────────────────────────
    model_path = args.model
    if not model_path:
        if args.model_type == "ask" and not args.setup_lanes:
            print("\n  [?] BẠN MUỐN CHẠY BẰNG MODEL NÀO?")
            print("      1. Model Fine-Tuning (Mặc định - models/weights/best.pt)")
            print("      2. Model From Scratch (scratch/Yolo_from_scratch/best.pt)")
            try:
                choice = input("  Nhập lựa chọn (1 hoặc 2) [Enter để chọn 1]: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  [Tự động] Bỏ qua chọn, sử dụng mặc định (1).")
                choice = "1"
            
            if choice == "2":
                model_path = "scratch/Yolo_from_scratch/best.pt"
            else:
                model_path = PIPELINE_CONFIG["model_path"]
        elif args.model_type == "scratch":
            model_path = "scratch/Yolo_from_scratch/best.pt"
        else:
            model_path = PIPELINE_CONFIG["model_path"]

    if args.benchmark:
        benchmark(model_path, source)
        return

    # ── Thiết lập Lane ROI tương tác ──────────────────────
    if args.setup_lanes:
        print("\n  CHẾ ĐỘ THIẾT LẬP ROI LÀN ĐƯỜNG")
        print("  ─────────────────────────────────")
        roi = LaneROI(n_lanes=args.lanes)
        saved = roi.setup_interactive(source=source)
        if saved:
            print(f"\n  ✅ ROI đã lưu — chạy lại pipeline bình thường:")
            print(f"     python main.py --source {source}")
        return

    cfg = {**PIPELINE_CONFIG,
           "target_fps"      : args.fps,
           "lane_n_lanes"    : args.lanes,
           "lane_ego_only"   : args.ego_only}

    pipeline = CollisionWarningPipeline(
        source       = source,
        model_path   = model_path,
        confidence   = args.conf,
        save_output  = args.save,
        enable_audio = not args.no_audio,
        show_window  = not args.no_window,
        config       = cfg,
    )

    print(f"\n  Phim tat khi chay:")
    print(f"    Q / ESC : Thoat")
    print(f"    S       : Chup anh snapshot")
    print(f"    P       : Tam dung (nhan phim bat ky de tiep)")
    print(f"\n  Lane ROI:")
    print(f"    Thiet lap lan dau: python main.py --setup-lanes --source {args.source}")

    pipeline.run()


if __name__ == "__main__":
    main()

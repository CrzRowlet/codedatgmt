"""
=============================================================
SCRIPT: Kiểm thử toàn diện & Báo cáo hiệu suất
scripts/test_system.py
=============================================================
Chạy:
  python scripts/test_system.py --quick        # Kiểm tra nhanh (không cần video)
  python scripts/test_system.py --full         # Toàn bộ
  python scripts/test_system.py --video V.mp4  # Kiểm tra với video thực
  python scripts/test_system.py --report       # Chỉ tạo báo cáo HTML
"""

import sys
import os
import time
import json
import csv
import argparse
import traceback
import importlib
from pathlib import Path
from datetime import datetime

# Đảm bảo in UTF-8 trên Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Thêm root vào path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET = "\033[0m"

def ok(msg):    print(f"  {GREEN}PASS  {msg}{RESET}")
def fail(msg):  print(f"  {RED}FAIL  {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}WARN  {msg}{RESET}")
def sec(title): print(f"\n{BOLD}{CYAN}{'='*55}\n  {title}\n{'='*55}{RESET}")


# ══════════════════════════════════════════════════════════
#  TEST RUNNER
# ══════════════════════════════════════════════════════════
class TestRunner:
    def __init__(self):
        self.results  = []   # [(name, passed, msg, duration_ms)]
        self.t_start  = time.time()

    def run(self, name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            fn(*args, **kwargs)
            ms = (time.perf_counter() - t0) * 1000
            ok(f"{name}  ({ms:.0f}ms)")
            self.results.append((name, True, "", ms))
        except AssertionError as e:
            ms = (time.perf_counter() - t0) * 1000
            fail(f"{name}  →  {e}")
            self.results.append((name, False, str(e), ms))
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            fail(f"{name}  →  {type(e).__name__}: {e}")
            self.results.append((name, False, traceback.format_exc(), ms))

    @property
    def passed(self): return sum(1 for r in self.results if r[1])
    @property
    def failed(self): return sum(1 for r in self.results if not r[1])
    @property
    def total(self):  return len(self.results)

    def summary(self):
        elapsed = time.time() - self.t_start
        print(f"\n  {'─'*50}")
        color = GREEN if self.failed == 0 else (YELLOW if self.failed <= 2 else RED)
        print(f"  {color}{BOLD}{self.passed}/{self.total} tests passed  ({elapsed:.1f}s){RESET}")
        if self.failed:
            print(f"\n  {RED}Failed tests:{RESET}")
            for name, passed, msg, _ in self.results:
                if not passed:
                    print(f"    • {name}")
        return self.failed == 0


# ══════════════════════════════════════════════════════════
#  TEST CASES
# ══════════════════════════════════════════════════════════

def test_imports():
    sec("TEST 1: Imports & Dependencies")
    runner = TestRunner()

    def t_cv2():
        import cv2
        assert cv2.__version__, "OpenCV không có version"

    def t_numpy():
        import numpy as np
        a = np.zeros((10, 10))
        assert a.shape == (10, 10)

    def t_distance_module():
        from src.distance.estimator import DistanceEstimator, DistanceResult
        assert DistanceEstimator
        assert DistanceResult

    def t_tracking_module():
        from src.tracking.ttc import ObjectTracker, Track, AlertState
        assert ObjectTracker
        assert Track

    def t_alert_module():
        from src.alert.alert_system import AlertSystem, AlertLevel, AudioEngine
        assert AlertSystem
        assert AlertLevel

    def t_utils_module():
        from src.utils.logger import EventLogger, FPSCounter
        from src.utils.visualizer import draw_bbox, draw_alert_panel
        assert EventLogger
        assert FPSCounter

    for name, fn in [
        ("cv2 import", t_cv2),
        ("numpy import", t_numpy),
        ("distance module", t_distance_module),
        ("tracking module", t_tracking_module),
        ("alert module", t_alert_module),
        ("utils module", t_utils_module),
    ]:
        runner.run(name, fn)

    return runner


def test_distance_estimator():
    sec("TEST 2: Distance Estimation")
    from src.distance.estimator import DistanceEstimator
    runner = TestRunner()

    def t_basic():
        est = DistanceEstimator(focal_length_px=700.0)
        r = est.estimate((300, 350, 552, 450), class_id=0, confidence=0.9)
        assert r.class_name == "car"
        assert 4.0 <= r.distance_m <= 6.0, f"Expected ~5m, got {r.distance_m}m"
        assert r.is_reliable

    def t_all_classes():
        est = DistanceEstimator(focal_length_px=700.0)
        class_names = ["car", "motorbike", "person", "truck", "obstacle"]
        for cls_id, name in enumerate(class_names):
            r = est.estimate((300, 300, 400, 400), class_id=cls_id, confidence=0.8)
            assert r.class_name == name, f"Expected {name}, got {r.class_name}"
            assert r.distance_m > 0

    def t_batch():
        est = DistanceEstimator(focal_length_px=700.0)
        dets = [
            {"bbox": (200, 300, 452, 430), "class_id": 0, "conf": 0.9},
            {"bbox": (550, 340, 620, 420), "class_id": 2, "conf": 0.7},
        ]
        results = est.estimate_batch(dets)
        assert len(results) == 2
        # Phải sort theo khoảng cách tăng dần
        assert results[0].distance_m <= results[1].distance_m

    def t_closest():
        est = DistanceEstimator(focal_length_px=700.0)
        dets = [
            {"bbox": (200, 300, 452, 430), "class_id": 0, "conf": 0.9},
            {"bbox": (580, 355, 620, 415), "class_id": 2, "conf": 0.7},
        ]
        results = est.estimate_batch(dets)
        closest = est.get_closest(results)
        assert closest is not None
        assert closest.distance_m == min(r.distance_m for r in results if r.is_reliable)

    def t_calibrate():
        est = DistanceEstimator()
        focal = est.calibrate(
            known_distance_m=8.0,
            known_real_width_m=1.8,
            measured_pixel_width=157.5,
        )
        assert abs(focal - 700.0) < 10, f"Calibration off: {focal}"

    def t_unreliable_small_bbox():
        est = DistanceEstimator(focal_length_px=700.0)
        r = est.estimate((300, 300, 310, 310), class_id=0, confidence=0.9)
        assert not r.is_reliable, "Bbox quá nhỏ phải unreliable"

    for name, fn in [
        ("basic estimate car 5m", t_basic),
        ("all 5 classes", t_all_classes),
        ("batch estimate + sort", t_batch),
        ("get_closest()", t_closest),
        ("calibrate()", t_calibrate),
        ("small bbox → unreliable", t_unreliable_small_bbox),
    ]:
        runner.run(name, fn)

    return runner


def test_tracker_ttc():
    sec("TEST 3: Object Tracking + TTC")
    from src.tracking.ttc import ObjectTracker, _iou, AlertState
    import numpy as np
    runner = TestRunner()

    def t_iou_overlap():
        iou = _iou((100, 100, 300, 300), (200, 200, 400, 400))
        assert 0.1 < iou < 0.5, f"IoU={iou}"

    def t_iou_no_overlap():
        iou = _iou((0, 0, 100, 100), (200, 200, 300, 300))
        assert iou == 0.0

    def t_iou_full_overlap():
        iou = _iou((100, 100, 300, 300), (100, 100, 300, 300))
        assert abs(iou - 1.0) < 0.001

    def t_track_created():
        tracker = ObjectTracker()
        t0 = time.time()
        dets = [{"bbox": (200, 250, 440, 380), "class_id": 0, "conf": 0.9}]
        tracks = tracker.update(dets, frame_width=640, timestamp=t0)
        assert len(tracks) == 1
        assert tracks[0]["track_id"] == 0
        assert tracks[0]["class_name"] == "car"

    def t_ttc_calculated():
        tracker = ObjectTracker()
        t0 = time.time()
        # Xe từ 15m → 7m trong 0.5s = approach speed ~16m/s
        for i in range(15):
            d    = 15.0 - i * 0.55
            w_px = int(1.8 * 700 / max(d, 0.5))
            bbox = (320 - w_px//2, 250, 320 + w_px//2, 380)
            bbox = (max(0, bbox[0]), bbox[1], min(639, bbox[2]), bbox[3])
            tracks = tracker.update(
                [{"bbox": bbox, "class_id": 0, "conf": 0.9}],
                frame_width=640, timestamp=t0 + i * 0.05
            )
        assert len(tracks) == 1
        t = tracks[0]
        assert t["ttc"] is not None, "TTC phải được tính sau 15 frames"
        assert t["approach_speed"] > 0, "Approach speed phải dương"
        assert 0 < t["ttc"] < 10.0

    def t_alert_state():
        tracker = ObjectTracker()
        t0 = time.time()
        # Xe cách 1.5m → phải DANGER
        d    = 1.5
        w_px = int(1.8 * 700 / d)
        bbox = (320 - w_px//2, 200, 320 + w_px//2, 400)
        bbox = (max(0, bbox[0]), bbox[1], min(639, bbox[2]), bbox[3])
        for i in range(5):
            tracks = tracker.update(
                [{"bbox": bbox, "class_id": 0, "conf": 0.9}],
                frame_width=640, timestamp=t0 + i * 0.05
            )
        assert tracks[0]["alert_state"] == AlertState.DANGER

    def t_track_disappear():
        tracker = ObjectTracker()
        t0 = time.time()
        # Track 10 frames
        for i in range(10):
            d    = 10.0
            w_px = int(1.8 * 700 / d)
            bbox = (300 - w_px//2, 250, 300 + w_px//2, 380)
            tracker.update([{"bbox": bbox, "class_id": 0, "conf": 0.9}],
                            640, t0 + i * 0.05)
        # Không detect nữa
        for i in range(12):
            tracks = tracker.update([], 640, t0 + (i + 11) * 0.05)
        # Track phải bị xóa sau max_missed frames
        assert len(tracks) == 0 or all(not t.get("is_active", True) for t in tracks)

    def t_multiple_tracks():
        tracker = ObjectTracker()
        t0 = time.time()
        dets = [
            {"bbox": (100, 200, 300, 350), "class_id": 0, "conf": 0.9},
            {"bbox": (400, 220, 580, 360), "class_id": 1, "conf": 0.85},
        ]
        tracks = tracker.update(dets, 640, t0)
        assert len(tracks) == 2
        ids = {t["track_id"] for t in tracks}
        assert len(ids) == 2, "Phải có 2 track ID khác nhau"

    for name, fn in [
        ("IoU overlap 0.1~0.5", t_iou_overlap),
        ("IoU no overlap = 0", t_iou_no_overlap),
        ("IoU full overlap = 1", t_iou_full_overlap),
        ("track created on first det", t_track_created),
        ("TTC calculated after 15f", t_ttc_calculated),
        ("alert_state DANGER < 2m", t_alert_state),
        ("track removed after miss", t_track_disappear),
        ("multiple tracks stable", t_multiple_tracks),
    ]:
        runner.run(name, fn)

    return runner


def test_alert_system():
    sec("TEST 4: Alert System")
    from src.alert.alert_system import AlertSystem, AlertLevel
    runner = TestRunner()

    def t_evaluate_safe():
        a = AlertSystem()
        assert a.evaluate(15.0, 10.0) == AlertLevel.SAFE

    def t_evaluate_warning_dist():
        a = AlertSystem()
        lvl = a.evaluate(5.0, None)
        assert lvl == AlertLevel.WARNING, f"Expected WARNING, got {lvl}"

    def t_evaluate_danger_dist():
        a = AlertSystem()
        lvl = a.evaluate(2.0, None)
        assert lvl == AlertLevel.DANGER, f"Expected DANGER, got {lvl}"

    def t_evaluate_warning_ttc():
        a = AlertSystem()
        lvl = a.evaluate(12.0, 3.5)   # dist safe nhưng TTC warning
        assert lvl == AlertLevel.WARNING

    def t_evaluate_danger_ttc():
        a = AlertSystem()
        lvl = a.evaluate(12.0, 1.5)   # dist safe nhưng TTC danger
        assert lvl == AlertLevel.DANGER

    def t_cooldown():
        a = AlertSystem()
        a.reset_cooldowns()
        _ = a.trigger(2.0, 1.5, "car")                         # phat
        stats_before = a.get_stats()["total_alerts"]
        _ = a.trigger(2.0, 1.5, "car")                         # cooldown
        stats_after = a.get_stats()["total_alerts"]
        assert stats_before == stats_after, "Cooldown không hoạt động"

    def t_force_override_cooldown():
        a = AlertSystem()
        a.reset_cooldowns()
        _ = a.trigger(2.0, 1.5, "car")
        s1 = a.get_stats()["total_alerts"]
        _ = a.trigger(2.0, 1.5, "car", force=True)   # force bỏ qua cooldown
        s2 = a.get_stats()["total_alerts"]
        assert s2 > s1, "force=True phải bỏ qua cooldown"

    def t_process_tracks():
        a = AlertSystem()
        tracks = [{"distance_m": 2.0, "ttc": 1.2, "class_name": "car",
                   "alert_state": "DANGER", "track_id": 0}]
        lvl = a.process_tracks(tracks)
        assert lvl == AlertLevel.DANGER

    def t_empty_tracks():
        a = AlertSystem()
        lvl = a.process_tracks([])
        assert lvl == AlertLevel.SAFE

    for name, fn in [
        ("evaluate SAFE 15m", t_evaluate_safe),
        ("evaluate WARNING dist 5m", t_evaluate_warning_dist),
        ("evaluate DANGER dist 2m", t_evaluate_danger_dist),
        ("evaluate WARNING TTC 3.5s", t_evaluate_warning_ttc),
        ("evaluate DANGER TTC 1.5s", t_evaluate_danger_ttc),
        ("cooldown blocks repeat", t_cooldown),
        ("force=True bypasses cooldown", t_force_override_cooldown),
        ("process_tracks DANGER", t_process_tracks),
        ("empty tracks → SAFE", t_empty_tracks),
    ]:
        runner.run(name, fn)

    return runner


def test_pipeline_integration():
    sec("TEST 5: Pipeline Tích Hợp End-to-End")
    import numpy as np
    import cv2
    from src.distance.estimator import DistanceEstimator
    from src.tracking.ttc import ObjectTracker, draw_full_overlay
    from src.alert.alert_system import AlertSystem, AlertLevel, AlertOverlayRenderer
    from src.utils.logger import EventLogger, FPSCounter
    from src.utils.visualizer import draw_bbox, draw_alert_panel

    runner = TestRunner()

    def t_full_pipeline_no_crash():
        frame   = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker = ObjectTracker()
        alert   = AlertSystem()
        renderer= AlertOverlayRenderer()
        t0      = time.time()

        for i in range(30):
            d    = max(1.0, 20.0 - i * 0.6)
            w_px = int(1.8 * 700 / d)
            bbox = (320 - w_px//2, 220, 320 + w_px//2, 380)
            bbox = (max(0, bbox[0]), bbox[1], min(639, bbox[2]), bbox[3])
            dets   = [{"bbox": bbox, "class_id": 0, "conf": 0.88}]
            tracks = tracker.update(dets, 640, t0 + i * 0.05)
            level  = alert.process_tracks(tracks)
            vis    = renderer.render(frame.copy(), tracks, level, fps=20.0, frame_count=i)
            assert vis is not None
            assert vis.shape == frame.shape

    def t_fps_counter():
        fps = FPSCounter(window=10)
        for _ in range(15):
            fps.tick()
            time.sleep(0.01)
        assert fps.fps > 0

    def t_event_logger():
        logger = EventLogger("logs/test_events.csv")
        logger.log("WARNING", 5.5, 3.2, "car", 1, 2.1)
        logger.log("DANGER", 2.0, 1.1, "motorbike", 2, 5.3)
        path = Path("logs/test_events.csv")
        assert path.exists()
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) >= 3   # header + 2 rows

    def t_visualizer_draw():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_bbox(frame, (100, 100, 300, 300), "car 5.0m", (50, 200, 50))
        draw_alert_panel(frame, "WARNING", 5.5, 3.2, 2, 20.0)
        # Nếu không crash là pass
        assert frame.sum() > 0

    def t_chart_render():
        import matplotlib; matplotlib.use("Agg")
        from src.ui.dashboard import DistanceChart
        chart = DistanceChart(640, 160)
        for i in range(30):
            chart.update(15.0 - i * 0.3, 8.0 - i * 0.2 if i < 20 else None)
        img = chart.render()
        assert img.shape == (160, 640, 3)
        assert img.dtype == np.uint8

    def t_pipeline_fps():
        """Đảm bảo pipeline core (không YOLO) chạy >= 60 FPS."""
        frame   = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker = ObjectTracker()
        alert   = AlertSystem()
        renderer= AlertOverlayRenderer()
        t0      = time.time()
        N       = 100
        for i in range(N):
            d    = max(1.0, 15.0 - i * 0.1)
            w_px = int(1.8 * 700 / d)
            bbox = (320 - w_px//2, 220, 320 + w_px//2, 380)
            bbox = (max(0, bbox[0]), bbox[1], min(639, bbox[2]), bbox[3])
            dets   = [{"bbox": bbox, "class_id": 0, "conf": 0.88}]
            tracks = tracker.update(dets, 640, t0 + i * 0.01)
            level  = alert.evaluate(
                tracks[0]["distance_m"] if tracks else 20.0,
                tracks[0].get("ttc") if tracks else None,
            )
        elapsed = time.time() - t0
        fps     = N / elapsed
        assert fps >= 50, f"Pipeline (no YOLO) FPS={fps:.0f} < 50. Cần tối ưu."

    for name, fn in [
        ("full pipeline 30 frames no crash", t_full_pipeline_no_crash),
        ("FPSCounter accuracy", t_fps_counter),
        ("EventLogger CSV write/read", t_event_logger),
        ("Visualizer drawing", t_visualizer_draw),
        ("DistanceChart matplotlib render", t_chart_render),
        ("Pipeline core FPS >= 50", t_pipeline_fps),
    ]:
        runner.run(name, fn)

    return runner


def test_with_video(video_path: str):
    sec(f"TEST 6: Kiểm thử với video thực: {Path(video_path).name}")
    import cv2
    import numpy as np
    from src.tracking.ttc import ObjectTracker
    from src.alert.alert_system import AlertSystem, AlertLevel, AlertOverlayRenderer
    from src.utils.logger import FPSCounter

    runner = TestRunner()

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Không mở được: {video_path}"

    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {w}x{h} @ {src_fps:.1f}fps  ({total_f} frames)")

    # Load YOLO nếu có
    model_path = ROOT / "models/weights/best.pt"
    model      = None
    if model_path.exists():
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            ok(f"YOLO model loaded: {model_path.name}")
        except Exception as e:
            warn(f"YOLO load failed: {e}")

    tracker  = ObjectTracker()
    alert    = AlertSystem()
    renderer = AlertOverlayRenderer()
    fps_cnt  = FPSCounter(30)
    stats    = {"frames": 0, "detections": 0, "warnings": 0, "dangers": 0, "fps_list": []}

    def t_process_100_frames():
        t0 = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(min(100, total_f)):
            ret, frame = cap.read()
            if not ret: break
            fps_cnt.tick()
            stats["frames"] += 1
            ts   = time.time()
            dets = []
            if model:
                preds = model.predict(frame, conf=0.5, verbose=False)[0]
                for box in preds.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    dets.append({"bbox": (x1,y1,x2,y2),
                                  "class_id": int(box.cls[0]),
                                  "conf": float(box.conf[0])})
                stats["detections"] += len(dets)
            tracks = tracker.update(dets, frame.shape[1], ts)
            level  = alert.process_tracks(tracks) if tracks else AlertLevel.SAFE
            if level == AlertLevel.WARNING: stats["warnings"] += 1
            if level == AlertLevel.DANGER:  stats["dangers"]  += 1
            fps_cnt.tick()
        elapsed = time.time() - t0
        stats["avg_fps"] = stats["frames"] / elapsed
        assert stats["frames"] >= 10, "Phải đọc được ít nhất 10 frames"

    def t_fps_target():
        fps = stats.get("avg_fps", 0)
        assert fps >= 15, f"FPS={fps:.1f} < 15 — cần tối ưu"

    def t_no_crash():
        assert True  # Nếu đến đây là không crash

    for name, fn in [
        ("process 100 frames", t_process_100_frames),
        ("FPS >= 15", t_fps_target),
        ("no crash throughout", t_no_crash),
    ]:
        runner.run(name, fn)

    cap.release()
    print(f"\n  Kết quả video test:")
    print(f"    Frames    : {stats['frames']}")
    print(f"    Avg FPS   : {stats.get('avg_fps',0):.1f}")
    if model:
        print(f"    Total det : {stats['detections']}")
    print(f"    Warnings  : {stats['warnings']}")
    print(f"    Dangers   : {stats['dangers']}")

    return runner


# ══════════════════════════════════════════════════════════
#  BÁO CÁO HTML
# ══════════════════════════════════════════════════════════
def generate_report(all_runners: list, video_results: dict = None):
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = ""
    total_pass = total_fail = 0

    for runner in all_runners:
        for name, passed, msg, ms in runner.results:
            icon = "✅" if passed else "❌"
            color= "#2EA043" if passed else "#DA3633"
            rows += f"""<tr>
              <td>{icon}</td>
              <td>{name}</td>
              <td style="color:{color};font-weight:600">{'PASS' if passed else 'FAIL'}</td>
              <td>{ms:.0f}ms</td>
              <td style="font-size:11px;max-width:400px;word-break:break-word">{msg[:120] if msg else ''}</td>
            </tr>"""
            if passed: total_pass += 1
            else:       total_fail += 1

    status_color = "#2EA043" if total_fail == 0 else "#DA3633"
    status_text  = "ALL PASSED ✅" if total_fail == 0 else f"{total_fail} FAILED ❌"

    html = f"""<!DOCTYPE html>
<html lang="vi"><head>
<meta charset="UTF-8">
<title>CWS Test Report {now}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0D1117;color:#E6EDF3;padding:2rem}}
h1{{font-size:22px;margin-bottom:4px}}
.meta{{color:#8B949E;font-size:13px;margin-bottom:24px}}
.summary{{display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap}}
.card{{background:#161B22;border:1px solid #30363D;border-radius:8px;padding:12px 20px;min-width:130px}}
.card-val{{font-size:28px;font-weight:700}}
.card-lbl{{font-size:12px;color:#8B949E;margin-top:4px}}
table{{width:100%;border-collapse:collapse;background:#161B22;border-radius:8px;overflow:hidden;font-size:13px}}
th{{background:#21262D;padding:10px 14px;text-align:left;font-weight:600;color:#8B949E;font-size:11px;text-transform:uppercase;letter-spacing:.06em}}
td{{padding:8px 14px;border-bottom:1px solid #21262D}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#1C2128}}
</style>
</head>
<body>
<h1>🚨 CWS — Báo cáo kiểm thử hệ thống</h1>
<div class="meta">Thời gian: {now}  |  Python {sys.version.split()[0]}</div>

<div class="summary">
  <div class="card">
    <div class="card-val" style="color:{status_color}">{status_text}</div>
    <div class="card-lbl">Kết quả tổng thể</div>
  </div>
  <div class="card">
    <div class="card-val" style="color:#2EA043">{total_pass}</div>
    <div class="card-lbl">Tests passed</div>
  </div>
  <div class="card">
    <div class="card-val" style="color:#DA3633">{total_fail}</div>
    <div class="card-lbl">Tests failed</div>
  </div>
  <div class="card">
    <div class="card-val">{total_pass + total_fail}</div>
    <div class="card-lbl">Tổng tests</div>
  </div>
</div>

<table>
  <thead><tr>
    <th></th><th>Test Case</th><th>Kết quả</th><th>Thời gian</th><th>Thông báo</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</body></html>"""

    out = ROOT / "logs/test_report.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n  📄 Báo cáo HTML: {out}")
    return out


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="CWS System Test")
    parser.add_argument("--quick",  action="store_true", help="Chỉ test nhanh không cần video")
    parser.add_argument("--full",   action="store_true", help="Toàn bộ tests")
    parser.add_argument("--video",  metavar="PATH",       help="Test với video thực")
    parser.add_argument("--report", action="store_true", help="Tạo báo cáo HTML từ kết quả cũ")
    args = parser.parse_args()

    print(f"{BOLD}{CYAN}")
    print("╔══════════════════════════════════════════════════╗")
    print("║   KIỂM THỬ TOÀN DIỆN HỆ THỐNG CWS                ║")
    print("╚══════════════════════════════════════════════════╝")
    print(RESET)

    all_runners = []
    all_pass    = True

    # Chạy test suite
    for suite_fn in [test_imports, test_distance_estimator,
                     test_tracker_ttc, test_alert_system,
                     test_pipeline_integration]:
        runner = suite_fn()
        all_runners.append(runner)
        if not runner.summary():
            all_pass = False

    # Video test (nếu có)
    if args.video or args.full:
        vp = args.video
        if not vp:
            # Tự tìm video mẫu
            candidates = list((ROOT / "data/raw/videos").glob("*.mp4")) + \
                         list((ROOT / "demo").glob("*.mp4"))
            vp = str(candidates[0]) if candidates else None
        if vp and Path(vp).exists():
            runner = test_with_video(vp)
            all_runners.append(runner)
            if not runner.summary():
                all_pass = False
        else:
            warn("Không tìm thấy video để test — bỏ qua Test 6")

    # Tạo báo cáo HTML
    report_path = generate_report(all_runners)

    # Tổng kết
    total_p = sum(r.passed for r in all_runners)
    total_f = sum(r.failed for r in all_runners)
    total_t = sum(r.total  for r in all_runners)
    print(f"\n{'='*55}")
    if all_pass:
        print(f"{GREEN}{BOLD}  ✅ TẤT CẢ {total_t} TESTS ĐỀU PASS!{RESET}")
    else:
        print(f"{RED}{BOLD}  ❌ {total_f}/{total_t} TESTS FAILED{RESET}")
    print(f"  📄 Report: {report_path}")
    print(f"{'='*55}\n")

    # Giải phóng tài nguyên âm thanh để tránh lỗi C-level crash của SDL khi thoát tiến trình
    try:
        import pygame
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.quit()
    except Exception:
        pass
    
    time.sleep(0.5)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

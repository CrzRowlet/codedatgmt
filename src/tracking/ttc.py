"""
=============================================================
MODULE: Object Tracking + TTC Calculation
src/tracking/ttc.py
=============================================================
Công thức TTC (Time-To-Collision):
    TTC = current_distance / approach_speed
    approach_speed = Δdistance / Δtime  (dương khi đang lại gần)

Cách dùng:
    python src/tracking/ttc.py --demo       # Demo với dữ liệu giả
    python src/tracking/ttc.py --test VIDEO # Test với video thực + YOLO
"""

import cv2
import numpy as np
import collections
import time
import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Import module khoảng cách
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.distance.estimator import DistanceEstimator, DistanceResult, REAL_WIDTHS


# ══════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════
HISTORY_SIZE        = 15     # Số frame để tính vận tốc trung bình
MIN_APPROACH_SPEED  = 0.05   # m/s — bỏ qua nếu tiếp cận chậm hơn
MIN_TRACK_FRAMES    = 3      # Phải theo dõi ít nhất N frame mới tính TTC
MAX_TTC_S           = 30.0   # Giới hạn TTC hiển thị (giây)
IOU_THRESHOLD       = 0.30   # Ngưỡng IoU để ghép track với detection mới

# Ngưỡng cảnh báo TTC (giây)
TTC_SAFE_THRESHOLD    = 5.0
TTC_WARNING_THRESHOLD = 2.0

# Khoảng cách cảnh báo tuyệt đối (mét) — dùng khi TTC không tin cậy
DIST_SAFE    = 10.0
DIST_WARNING =  5.0
DIST_DANGER  =  2.0


# ══════════════════════════════════════════════════════════
#  TRẠNG THÁI CẢNH BÁO
# ══════════════════════════════════════════════════════════
class AlertState:
    SAFE    = "SAFE"
    WARNING = "WARNING"
    DANGER  = "DANGER"

ALERT_COLORS = {
    AlertState.SAFE    : (50,  200, 50),    # BGR xanh lá
    AlertState.WARNING : (0,   165, 255),   # cam
    AlertState.DANGER  : (0,   0,   230),   # đỏ
}
ALERT_LABELS = {
    AlertState.SAFE    : "AN TOAN",
    AlertState.WARNING : "CANH BAO",
    AlertState.DANGER  : "NGUY HIEM",
}


# ══════════════════════════════════════════════════════════
#  DATACLASS: Thông tin một track
# ══════════════════════════════════════════════════════════
@dataclass
class Track:
    track_id    : int
    class_id    : int
    class_name  : str
    # Lịch sử (timestamp, khoảng cách)
    dist_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=HISTORY_SIZE))
    time_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=HISTORY_SIZE))
    last_bbox   : Tuple[int,int,int,int] = (0,0,0,0)
    age         : int = 0        # Số frame đã theo dõi
    missed      : int = 0        # Frame liên tiếp không detect được
    is_active   : bool = True

    def update(self, distance_m: float, bbox: Tuple[int,int,int,int], timestamp: float):
        self.dist_history.append(distance_m)
        self.time_history.append(timestamp)
        self.last_bbox = bbox
        self.age += 1
        self.missed = 0

    @property
    def current_distance(self) -> Optional[float]:
        return self.dist_history[-1] if self.dist_history else None

    @property
    def approach_speed(self) -> float:
        """
        Tính vận tốc tiếp cận (m/s) bằng linear regression.
        Giá trị dương = đang lại gần, âm = đang rời xa.
        """
        if len(self.dist_history) < 2:
            return 0.0

        dists  = np.array(list(self.dist_history))
        times  = np.array(list(self.time_history))
        dt     = times[-1] - times[0]

        if dt < 0.01:
            return 0.0

        # Linear regression: d(t) = a*t + b → slope a = approach speed (âm khi lại gần)
        if len(dists) >= 3:
            coef = np.polyfit(times - times[0], dists, 1)
            slope = coef[0]   # m/s — âm khi khoảng cách giảm (đang lại gần)
        else:
            slope = (dists[-1] - dists[0]) / dt

        return -slope   # Đổi dấu: dương = đang tiếp cận

    @property
    def ttc(self) -> Optional[float]:
        """
        Tính TTC (giây).
        Trả về None nếu không đủ dữ liệu hoặc vật đang rời xa.
        """
        if self.age < MIN_TRACK_FRAMES or not self.dist_history:
            return None

        speed = self.approach_speed
        dist  = self.current_distance

        if speed < MIN_APPROACH_SPEED or dist is None:
            return None   # Đứng yên hoặc đang rời xa

        ttc_val = dist / speed
        return min(ttc_val, MAX_TTC_S)

    @property
    def alert_state(self) -> str:
        """Xác định mức cảnh báo dựa trên TTC + khoảng cách."""
        dist = self.current_distance
        ttc  = self.ttc

        if dist is None:
            return AlertState.SAFE

        # Ưu tiên khoảng cách tuyệt đối trước
        if dist <= DIST_DANGER:
            return AlertState.DANGER
        if dist <= DIST_WARNING:
            return AlertState.WARNING

        # Sau đó dùng TTC
        if ttc is not None:
            if ttc <= TTC_WARNING_THRESHOLD:
                return AlertState.DANGER
            if ttc <= TTC_SAFE_THRESHOLD:
                return AlertState.WARNING

        return AlertState.SAFE


# ══════════════════════════════════════════════════════════
#  CLASS CHÍNH: ObjectTracker
# ══════════════════════════════════════════════════════════
class ObjectTracker:
    """
    Theo dõi vật thể qua các frame và tính TTC.
    Dùng thuật toán centroid tracking + IoU matching (nhẹ, không cần thư viện ngoài).
    """

    def __init__(self, max_missed: int = 10):
        self.tracks     : Dict[int, Track] = {}
        self._next_id   : int = 0
        self.max_missed : int = max_missed   # Frame không detect → xóa track
        self.estimator  = DistanceEstimator()
        self._frame_time: float = 0.0

    # ── Cập nhật tracker với detections mới ──────────────
    def update(
        self,
        detections : List[dict],   # [{"bbox":(x1,y1,x2,y2), "class_id":int, "conf":float}]
        frame_width: int = 1280,
        timestamp  : float = None,
    ) -> List[dict]:
        """
        Cập nhật tracker một frame.

        Args:
            detections : Danh sách detection từ YOLO
            frame_width: Chiều rộng frame
            timestamp  : Thời điểm frame (giây)

        Returns:
            Danh sách track info với TTC và alert state
        """
        if timestamp is None:
            timestamp = time.time()
        self._frame_time = timestamp

        # Ước lượng khoảng cách cho tất cả detection
        dist_results = self.estimator.estimate_batch(detections, frame_width)

        # Map detection_idx → DistanceResult
        det_to_dist = {i: dr for i, dr in enumerate(dist_results)}

        if not self.tracks:
            # Frame đầu tiên: tạo track mới cho mọi detection
            for i, det in enumerate(detections):
                self._create_track(det, det_to_dist[i], timestamp)
        else:
            # Match detection với track hiện có bằng IoU
            matched, unmatched_dets, unmatched_tracks = self._match(detections)

            # Cập nhật track đã match
            for det_idx, track_id in matched:
                det  = detections[det_idx]
                dist = det_to_dist[det_idx]
                self.tracks[track_id].update(dist.distance_m, det["bbox"], timestamp)

            # Tạo track mới cho detection chưa match
            for det_idx in unmatched_dets:
                self._create_track(detections[det_idx], det_to_dist[det_idx], timestamp)

            # Tăng missed cho track không có detection
            for track_id in unmatched_tracks:
                self.tracks[track_id].missed += 1
                if self.tracks[track_id].missed > self.max_missed:
                    self.tracks[track_id].is_active = False

        # Dọn dẹp track không active
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.is_active}

        # Trả về thông tin đầy đủ
        return self._build_output()

    def _create_track(self, det: dict, dist_result: DistanceResult, timestamp: float):
        tid   = self._next_id
        self._next_id += 1
        track = Track(
            track_id   = tid,
            class_id   = det["class_id"],
            class_name = dist_result.class_name,
            last_bbox  = det["bbox"],
        )
        track.update(dist_result.distance_m, det["bbox"], timestamp)
        self.tracks[tid] = track

    def _match(self, detections: List[dict]) -> Tuple[List, List, List]:
        """
        Ghép detection mới với track cũ bằng IoU centroid.
        Returns: (matched_pairs, unmatched_det_indices, unmatched_track_ids)
        """
        if not detections:
            return [], [], list(self.tracks.keys())

        track_ids  = list(self.tracks.keys())
        det_bboxes = [d["bbox"] for d in detections]
        trk_bboxes = [self.tracks[tid].last_bbox for tid in track_ids]

        # Tính IoU matrix
        iou_mat = np.zeros((len(detections), len(track_ids)))
        for i, db in enumerate(det_bboxes):
            for j, tb in enumerate(trk_bboxes):
                iou_mat[i, j] = _iou(db, tb)

        # Greedy matching
        matched         = []
        used_dets       = set()
        used_trks       = set()

        # Sắp xếp theo IoU giảm dần
        pairs = sorted(
            [(i, j, iou_mat[i, j]) for i in range(len(detections)) for j in range(len(track_ids))],
            key=lambda x: -x[2]
        )
        for det_i, trk_j, iou_val in pairs:
            if iou_val < IOU_THRESHOLD: break
            if det_i in used_dets or trk_j in used_trks: continue
            matched.append((det_i, track_ids[trk_j]))
            used_dets.add(det_i)
            used_trks.add(trk_j)

        unmatched_dets  = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_trks  = [track_ids[j] for j in range(len(track_ids)) if j not in {p[1] for p in [(m[0], track_ids.index(m[1])) for m in matched]}]

        return matched, unmatched_dets, list(self.tracks.keys() - set(t for _, t in matched))

    def _build_output(self) -> List[dict]:
        """Xây dựng danh sách output trả về cho pipeline chính."""
        output = []
        for tid, track in sorted(self.tracks.items()):
            if not track.is_active or track.current_distance is None:
                continue
            entry = {
                "track_id"      : tid,
                "class_id"      : track.class_id,
                "class_name"    : track.class_name,
                "bbox"          : track.last_bbox,
                "distance_m"    : round(track.current_distance, 2),
                "approach_speed": round(track.approach_speed, 3),
                "ttc"           : round(track.ttc, 2) if track.ttc is not None else None,
                "alert_state"   : track.alert_state,
                "age_frames"    : track.age,
            }
            output.append(entry)
        # Sắp xếp: DANGER → WARNING → SAFE, rồi theo khoảng cách
        priority = {AlertState.DANGER: 0, AlertState.WARNING: 1, AlertState.SAFE: 2}
        output.sort(key=lambda x: (priority[x["alert_state"]], x["distance_m"]))
        return output

    def get_most_critical(self) -> Optional[dict]:
        """Trả về vật thể nguy hiểm nhất hiện tại."""
        out = self._build_output()
        return out[0] if out else None

    def reset(self):
        self.tracks    = {}
        self._next_id  = 0


def _iou(boxA, boxB) -> float:
    """Tính Intersection over Union của 2 bbox."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(aA + aB - inter)


# ══════════════════════════════════════════════════════════
#  HÀM VẼ OVERLAY TỔNG HỢP (Distance + TTC + Alert)
# ══════════════════════════════════════════════════════════
def draw_full_overlay(frame: np.ndarray, tracks_output: List[dict]) -> np.ndarray:
    """
    Vẽ toàn bộ thông tin lên frame: bbox, khoảng cách, TTC, alert state.
    Dùng trong pipeline tích hợp.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    for t in tracks_output:
        x1, y1, x2, y2 = t["bbox"]
        state  = t["alert_state"]
        color  = ALERT_COLORS[state]

        # Bounding box — dày hơn nếu nguy hiểm
        thick = 3 if state == AlertState.DANGER else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        # Góc bounding box style (điểm nhấn trực quan)
        l = min(15, (x2-x1)//4)
        for px, py, dx, dy in [(x1,y1,l,0),(x1,y1,0,l),(x2,y1,-l,0),(x2,y1,0,l),
                                 (x1,y2,l,0),(x1,y2,0,-l),(x2,y2,-l,0),(x2,y2,0,-l)]:
            cv2.line(out, (px,py), (px+dx, py+dy), color, 2)

        # Dòng 1: class + khoảng cách
        line1 = f"{t['class_name']}  {t['distance_m']:.1f}m"
        # Dòng 2: TTC
        if t["ttc"] is not None:
            line2 = f"TTC: {t['ttc']:.1f}s"
        else:
            speed = t["approach_speed"]
            line2 = f"spd: {speed:+.1f}m/s" if abs(speed) > 0.05 else "TTC: --"

        # Vẽ background label
        for k, line in enumerate([line1, line2]):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            lx = x1
            ly = y1 - 5 - k * (th + 6)
            ly = max(ly, th + 4)
            cv2.rectangle(out, (lx, ly-th-3), (lx+tw+6, ly+2), color, -1)
            cv2.putText(out, line, (lx+3, ly-1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # Track ID nhỏ
        cv2.putText(out, f"#{t['track_id']}", (x2+3, y1+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Panel trạng thái tổng hợp (góc trên trái)
    _draw_status_panel(out, tracks_output)
    return out


def _draw_status_panel(frame: np.ndarray, tracks: List[dict]):
    """Vẽ panel tổng hợp góc trên trái."""
    if not tracks:
        return

    most_critical = tracks[0]  # Đã sắp xếp: nguy hiểm nhất trước
    state  = most_critical["alert_state"]
    color  = ALERT_COLORS[state]
    label  = ALERT_LABELS[state]
    dist   = most_critical["distance_m"]
    ttc    = most_critical["ttc"]

    # Nền panel
    panel_w, panel_h = 240, 80
    overlay  = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8+panel_w, 8+panel_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Đường viền màu theo trạng thái
    cv2.rectangle(frame, (8, 8), (8+panel_w, 8+panel_h), color, 2)

    # Text
    cv2.putText(frame, label, (16, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Dist: {dist:.1f}m", (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)
    if ttc is not None:
        cv2.putText(frame, f"TTC:  {ttc:.1f}s", (130, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════
#  DEMO KHÔNG CẦN CAMERA/YOLO
# ══════════════════════════════════════════════════════════
def demo_simulation():
    """
    Mô phỏng xe đang tiến lại gần — minh họa TTC calculation.
    Không cần camera hay YOLO model.
    """
    print("\n  DEMO MÔ PHỎNG TTC CALCULATION")
    print("  " + "─"*60)
    print(f"  {'Frame':>5} {'Dist(m)':>10} {'Speed(m/s)':>12} {'TTC(s)':>10} {'Alert':>12}")
    print("  " + "─"*60)

    tracker = ObjectTracker()
    t0 = time.time()

    # Mô phỏng 60 frame, xe tiến từ 20m → 1m trong ~3 giây
    distances = np.linspace(20.0, 1.0, 60)
    fps_sim   = 20.0

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    for frame_i, target_dist in enumerate(distances):
        ts  = t0 + frame_i / fps_sim
        # Tạo bbox giả tỷ lệ nghịch với khoảng cách
        w_px = int(1.8 * 700 / target_dist)
        cx, cy = 320, 300
        bbox = (cx - w_px//2, cy - 40, cx + w_px//2, cy + 40)
        bbox = (max(0,bbox[0]), max(0,bbox[1]), min(639,bbox[2]), min(479,bbox[3]))

        det = [{"bbox": bbox, "class_id": 0, "conf": 0.9}]
        result = tracker.update(det, frame_width=640, timestamp=ts)

        if result:
            r = result[0]
            ttc_str   = f"{r['ttc']:.1f}" if r["ttc"] else "  --"
            speed_str = f"{r['approach_speed']:.2f}"
            print(f"  {frame_i+1:>5} {r['distance_m']:>10.1f} {speed_str:>12} {ttc_str:>10} {r['alert_state']:>12}")

            # Visualize
            canvas[:] = (25, 25, 35)
            tracks_out = result
            # Vẽ xe giả (hình chữ nhật)
            cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (100,100,100), 2)
            canvas = draw_full_overlay(canvas, tracks_out)

            cv2.imshow("TTC Demo — Simulation", canvas)
            key = cv2.waitKey(50)
            if key == ord("q"):
                break

    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    print("\n  GIẢI THÍCH:")
    print("  • Khi xe tiến từ 20m → 5m (TTC > 5s):  AN TOAN")
    print("  • Khi xe ở 5m (TTC ≈ 2–5s):             CANH BAO")
    print("  • Khi xe dưới 2m (TTC < 2s):            NGUY HIEM")


# ══════════════════════════════════════════════════════════
#  TEST VỚI VIDEO THỰC + YOLO
# ══════════════════════════════════════════════════════════
def test_with_video(video_path: str, model_path: str = "models/weights/best.pt"):
    """
    Chạy full pipeline: YOLO detect → distance → TTC → overlay.
    Dùng để kiểm thử với video thực.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(f"  Model: {model_path} ✅")
    except Exception as e:
        print(f"  Lỗi load model: {e}")
        print(f"  Chạy demo không cần model: python src/tracking/ttc.py --demo")
        return

    cap = cv2.VideoCapture(video_path if video_path != "0" else 0)
    if not cap.isOpened():
        print(f"  Không mở được: {video_path}")
        return

    tracker = ObjectTracker()
    fps_target = 20

    print(f"  Video: {video_path}")
    print("  Nhấn Q để thoát\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        t_start = time.time()

        # YOLO inference
        preds = model.predict(frame, conf=0.5, iou=0.45, verbose=False)[0]
        detections = []
        for box in preds.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox"     : (x1, y1, x2, y2),
                "class_id" : int(box.cls[0]),
                "conf"     : float(box.conf[0]),
            })

        # Tracker + TTC
        tracks = tracker.update(detections, frame_width=frame.shape[1])

        # Overlay
        vis = draw_full_overlay(frame, tracks)

        # FPS
        elapsed = time.time() - t_start
        fps_actual = 1.0 / max(elapsed, 1e-5)
        cv2.putText(vis, f"FPS: {fps_actual:.0f}", (vis.shape[1]-100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)

        cv2.imshow("TTC Test", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Tracking + TTC")
    parser.add_argument("--demo",  action="store_true",
                        help="Chạy demo mô phỏng (không cần camera/model)")
    parser.add_argument("--test",  metavar="VIDEO",
                        help="Test với video thực (0=webcam hoặc path file)")
    parser.add_argument("--model", default="models/weights/best.pt",
                        help="Đường dẫn model YOLO (.pt)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  OBJECT TRACKING + TTC CALCULATION")
    print(f"{'='*55}")

    if args.demo:
        demo_simulation()
    elif args.test:
        test_with_video(args.test, args.model)
    else:
        # Kiểm tra nhanh không cần gì
        print("\n  Unit Test — ObjectTracker + TTC:\n")
        tracker = ObjectTracker()
        t0 = time.time()

        # Mô phỏng 10 frame
        for i in range(10):
            d = 15.0 - i * 1.2          # Giảm dần từ 15m → 1.8m
            w_px = int(1.8 * 700 / d)
            bbox = (320-w_px//2, 260, 320+w_px//2, 340)
            dets = [{"bbox": bbox, "class_id": 0, "conf": 0.9}]
            out  = tracker.update(dets, frame_width=640, timestamp=t0 + i * 0.05)
            if out:
                r = out[0]
                ttc_s = f"{r['ttc']:.1f}s" if r["ttc"] else "N/A"
                print(f"  Frame {i+1:2d}: dist={r['distance_m']:.1f}m  "
                      f"speed={r['approach_speed']:.2f}m/s  TTC={ttc_s:>6}  [{r['alert_state']}]")

        print("\n  Cách dùng thực tế:")
        print("    python src/tracking/ttc.py --demo")
        print("    python src/tracking/ttc.py --test data/raw/videos/test.mp4 --model models/weights/best.pt")

"""
=============================================================
MODULE: Ước lượng khoảng cách (Distance Estimation)
src/distance/estimator.py
=============================================================
Phương pháp: Pinhole Camera Model
  distance = (real_width × focal_length_px) / pixel_width

Cách dùng standalone:
  python src/distance/estimator.py --calibrate   # Hiệu chỉnh camera
  python src/distance/estimator.py --test        # Test với webcam/video
  python src/distance/estimator.py --benchmark   # Đo sai số trên ảnh test
"""

import cv2
import numpy as np
import json
import os
import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Tuple, List

# Đảm bảo in UTF-8 trên Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


# ══════════════════════════════════════════════════════════
#  DATACLASS KẾT QUẢ
# ══════════════════════════════════════════════════════════
@dataclass
class DistanceResult:
    """Kết quả ước lượng khoảng cách cho một vật thể."""
    class_id   : int
    class_name : str
    bbox       : Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    pixel_width: float                        # Chiều rộng bbox tính bằng pixel
    distance_m : float                        # Khoảng cách ước lượng (mét)
    confidence : float                        # Confidence từ YOLO
    is_reliable: bool                         # True nếu bbox đủ lớn/tin cậy

    def to_dict(self):
        return asdict(self)


# ══════════════════════════════════════════════════════════
#  THÔNG SỐ VẬT LÝ VẬT THỂ (mét)
# ══════════════════════════════════════════════════════════
# Chiều rộng trung bình thực tế theo class
REAL_WIDTHS: Dict[str, float] = {
    "car"       : 1.80,   # Ô tô tiêu chuẩn
    "motorbike" : 0.70,   # Xe máy
    "person"    : 0.50,   # Người (vai)
    "truck"     : 2.50,   # Xe tải
}

# Ngưỡng pixel width tối thiểu để ước lượng tin cậy
MIN_PIXEL_WIDTH = 20   # px — bbox quá nhỏ sẽ gây sai số lớn
MAX_DISTANCE_M  = 50.0  # m  — quá xa thì không tin cậy

# File lưu thông số hiệu chỉnh camera
CALIB_FILE = "configs/camera_calib.json"

# Thông số mặc định (trước khi hiệu chỉnh)
DEFAULT_FOCAL_PX = 700.0   # pixel — ước tính cho camera 1080p thông thường


# ══════════════════════════════════════════════════════════
#  CLASS CHÍNH: DistanceEstimator
# ══════════════════════════════════════════════════════════
class DistanceEstimator:
    """
    Ước lượng khoảng cách đến vật thể dựa trên Pinhole Camera Model.

    Công thức cốt lõi:
        distance = (W_real × f) / W_pixel

    Trong đó:
        W_real  = chiều rộng thực của vật thể (mét)
        f       = tiêu cự camera (pixel) — hiệu chỉnh bằng calibrate()
        W_pixel = chiều rộng bounding box (pixel)
    """

    def __init__(
        self,
        focal_length_px : float = DEFAULT_FOCAL_PX,
        real_widths     : Dict[str, float] = None,
        class_names     : List[str] = None,
        calib_file      : str = CALIB_FILE,
    ):
        self.focal_length_px = focal_length_px
        self.real_widths     = real_widths or REAL_WIDTHS.copy()
        self.class_names     = class_names or list(REAL_WIDTHS.keys())
        self.calib_file      = calib_file
        self._calib_history  : List[dict] = []   # Lưu lịch sử hiệu chỉnh

        # Tải hiệu chỉnh đã lưu (nếu có)
        self._load_calibration()

    # ── Tải/Lưu hiệu chỉnh ───────────────────────────────
    def _load_calibration(self):
        if os.path.exists(self.calib_file):
            with open(self.calib_file) as f:
                data = json.load(f)
            self.focal_length_px = data.get("focal_length_px", DEFAULT_FOCAL_PX)
            print(f"[DistanceEstimator] Loaded calibration: focal={self.focal_length_px:.1f}px  ({self.calib_file})")
        else:
            print(f"[DistanceEstimator] Dùng focal mặc định: {self.focal_length_px}px  (chưa hiệu chỉnh)")

    def save_calibration(self):
        os.makedirs(os.path.dirname(self.calib_file), exist_ok=True)
        data = {
            "focal_length_px" : self.focal_length_px,
            "calibrated_at"   : str(np.datetime64("now")),
            "history"         : self._calib_history,
        }
        with open(self.calib_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DistanceEstimator] Saved: {self.calib_file}")

    # ── Hiệu chỉnh tiêu cự từ điểm đo thực tế ────────────
    def calibrate(
        self,
        known_distance_m   : float,
        known_real_width_m : float,
        measured_pixel_width: float,
    ) -> float:
        """
        Tính tiêu cự từ một điểm đo đã biết.

        Đặt camera hành trình, ghi lại xe ô tô ở khoảng cách D mét.
        Đo chiều rộng bbox trên ảnh (pixel_width).
        Gọi hàm này với các giá trị đó.

        Args:
            known_distance_m    : Khoảng cách thực đo được bằng thước/GPS (mét)
            known_real_width_m  : Chiều rộng thực của vật thể (mét), vd 1.8 cho ô tô
            measured_pixel_width: Chiều rộng bbox đo trên ảnh (pixel)

        Returns:
            float: Giá trị focal_length tính được (pixel)

        Công thức nghịch đảo:
            f = (W_pixel × distance) / W_real
        """
        focal = (measured_pixel_width * known_distance_m) / known_real_width_m
        self.focal_length_px = focal

        entry = {
            "distance_m"    : known_distance_m,
            "real_width_m"  : known_real_width_m,
            "pixel_width"   : measured_pixel_width,
            "focal_result"  : round(focal, 2),
        }
        self._calib_history.append(entry)

        print(f"[Calibrate] D={known_distance_m}m  W_real={known_real_width_m}m  "
              f"W_px={measured_pixel_width:.0f}px  →  focal={focal:.1f}px")
        return focal

    def calibrate_multi(self, measurements: List[dict]) -> float:
        """
        Hiệu chỉnh từ nhiều điểm đo, lấy trung bình.
        measurements: [{"distance_m": 5.0, "real_width_m": 1.8, "pixel_width": 252}, ...]
        """
        focals = []
        for m in measurements:
            f = self.calibrate(m["distance_m"], m["real_width_m"], m["pixel_width"])
            focals.append(f)
        avg = float(np.mean(focals))
        std = float(np.std(focals))
        self.focal_length_px = avg
        print(f"[Calibrate Multi] avg_focal={avg:.1f}  std={std:.1f}  (từ {len(focals)} điểm đo)")
        return avg

    # ── Ước lượng khoảng cách ─────────────────────────────
    def estimate(
        self,
        bbox       : Tuple[int, int, int, int],
        class_id   : int,
        confidence : float = 1.0,
        frame_width: int   = 1280,
    ) -> DistanceResult:
        """
        Ước lượng khoảng cách từ một bounding box.

        Args:
            bbox       : (x1, y1, x2, y2) — tọa độ pixel
            class_id   : Index class (dùng self.class_names để tra tên)
            confidence : Confidence score từ YOLO
            frame_width: Chiều rộng frame (để fallback nếu cần)

        Returns:
            DistanceResult
        """
        x1, y1, x2, y2 = bbox
        pixel_width = float(x2 - x1)

        # Lấy tên class và chiều rộng thực
        if 0 <= class_id < len(self.class_names):
            class_name = self.class_names[class_id]
        else:
            class_name = f"class_{class_id}"

        real_width = self.real_widths.get(class_name, 1.0)

        # Kiểm tra độ tin cậy
        is_reliable = pixel_width >= MIN_PIXEL_WIDTH and confidence >= 0.4

        # Tính khoảng cách
        if pixel_width > 0:
            distance = (real_width * self.focal_length_px) / pixel_width
            distance = min(distance, MAX_DISTANCE_M)
        else:
            distance = MAX_DISTANCE_M

        return DistanceResult(
            class_id    = class_id,
            class_name  = class_name,
            bbox        = bbox,
            pixel_width = pixel_width,
            distance_m  = round(distance, 2),
            confidence  = round(confidence, 3),
            is_reliable = is_reliable,
        )

    def estimate_batch(
        self,
        detections : List[dict],
        frame_width: int = 1280,
    ) -> List[DistanceResult]:
        """
        Ước lượng cho nhiều detection cùng lúc.
        detections: [{"bbox": (x1,y1,x2,y2), "class_id": int, "conf": float}, ...]
        """
        results = []
        for det in detections:
            r = self.estimate(
                bbox        = tuple(det["bbox"]),
                class_id    = det["class_id"],
                confidence  = det.get("conf", 1.0),
                frame_width = frame_width,
            )
            results.append(r)
        # Sắp xếp theo khoảng cách tăng dần (vật gần nhất trước)
        results.sort(key=lambda r: r.distance_m)
        return results

    def get_closest(self, results: List[DistanceResult]) -> Optional[DistanceResult]:
        """Trả về vật thể gần nhất (reliable)."""
        reliable = [r for r in results if r.is_reliable]
        return min(reliable, key=lambda r: r.distance_m) if reliable else None

    # ── Vẽ overlay lên frame ──────────────────────────────
    def draw_overlay(
        self,
        frame  : np.ndarray,
        results: List[DistanceResult],
        alert_distances: dict = None,   # {"warning": 5.0, "danger": 2.0}
    ) -> np.ndarray:
        """
        Vẽ bounding box và thông tin khoảng cách lên frame.
        Màu sắc:
            Xanh lá  → an toàn (> warning_dist)
            Vàng     → cảnh báo
            Đỏ       → nguy hiểm (< danger_dist)
        """
        frame_out = frame.copy()
        ad = alert_distances or {"warning": 5.0, "danger": 2.0}

        # Màu sắc per-class (BGR)
        CLASS_COLORS = {
            "car"       : (50,  200, 50),
            "motorbike" : (0,   165, 255),
            "person"    : (255, 100, 0),
            "truck"     : (200, 0,   200),
            "obstacle"  : (0,   0,   220),
        }

        for r in results:
            if not r.is_reliable:
                continue

            x1, y1, x2, y2 = r.bbox

            # Màu theo mức nguy hiểm
            if r.distance_m <= ad["danger"]:
                color    = (0, 0, 230)     # Đỏ — NGUY HIỂM
                label_bg = (0, 0, 180)
            elif r.distance_m <= ad["warning"]:
                color    = (0, 165, 255)   # Cam — CẢNH BÁO
                label_bg = (0, 120, 200)
            else:
                color    = CLASS_COLORS.get(r.class_name, (50, 200, 50))  # Xanh — AN TOÀN
                label_bg = (20, 140, 30)

            # Vẽ bounding box
            thickness = 3 if r.distance_m <= ad["danger"] else 2
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, thickness)

            # Label text
            label = f"{r.class_name}  {r.distance_m:.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lx, ly = x1, max(y1 - 5, th + 4)

            # Nền label
            cv2.rectangle(frame_out, (lx, ly - th - 4), (lx + tw + 6, ly + 2), label_bg, -1)
            cv2.putText(frame_out, label, (lx + 3, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return frame_out


# ══════════════════════════════════════════════════════════
#  HIỆU CHỈNH TƯƠNG TÁC
# ══════════════════════════════════════════════════════════
def interactive_calibration():
    """
    Hướng dẫn hiệu chỉnh tiêu cự camera hành trình thực tế.
    Dùng khi có video thực + thước đo.
    """
    print("""
╔══════════════════════════════════════════════════════╗
║   HƯỚNG DẪN HIỆU CHỈNH CAMERA HÀNH TRÌNH           ║
╚══════════════════════════════════════════════════════╝

Bạn cần:
  - 1 video thực từ camera hành trình
  - Biết khoảng cách thực đến một xe ô tô trong clip
    (dùng GPS, thước dây hoặc odometer)
  - Đo chiều rộng bbox của xe đó trên ảnh

Cách làm:
  1. Mở video bằng VLC hoặc OpenCV
  2. Pause ở frame có xe ô tô rõ ràng, cách ~5–10m
  3. Đo pixel_width của bbox xe (dùng paint / GIMP / imshow)
  4. Nhập các giá trị vào bên dưới
""")

    est = DistanceEstimator()
    measurements = []

    print("Nhập các điểm đo hiệu chỉnh (enter để kết thúc):\n")
    while True:
        try:
            dist  = input("  Khoảng cách thực (mét, vd 8.0):  ").strip()
            if not dist: break
            width = input("  Chiều rộng bbox (pixel, vd 225):  ").strip()
            cls   = input("  Class (car/truck/person, mặc định car): ").strip() or "car"
            real_w = REAL_WIDTHS.get(cls, 1.8)
            measurements.append({
                "distance_m"   : float(dist),
                "real_width_m" : real_w,
                "pixel_width"  : float(width),
            })
            print(f"  ✓ Đã thêm điểm đo #{len(measurements)}\n")
        except (ValueError, KeyboardInterrupt):
            break

    if measurements:
        focal = est.calibrate_multi(measurements)
        est.save_calibration()
        print(f"\n  ✅ Focal = {focal:.1f}px — đã lưu vào {CALIB_FILE}")
        print(f"  Kiểm tra nhanh:")
        test_widths = [50, 100, 200, 350, 500]
        print(f"  {'BBox width (px)':<20} {'Khoảng cách ước lượng':>25}")
        for px in test_widths:
            d = (1.8 * focal) / px
            print(f"  {px:<20} {d:>20.2f} m  (xe ô tô)")
    else:
        print("  Không có điểm đo. Dùng focal mặc định.")


# ══════════════════════════════════════════════════════════
#  BENCHMARK ĐỘ CHÍNH XÁC
# ══════════════════════════════════════════════════════════
def benchmark_accuracy(ground_truth_file: str):
    """
    Đo sai số ước lượng khoảng cách so với ground truth.

    Format file JSON:
    [{"bbox": [x1,y1,x2,y2], "class_id": 0, "conf": 0.9, "true_distance_m": 5.2}, ...]
    """
    if not os.path.exists(ground_truth_file):
        # Tạo file mẫu
        sample = [
            {"bbox": [400, 350, 625, 480], "class_id": 0, "conf": 0.92, "true_distance_m": 5.0},
            {"bbox": [550, 360, 720, 470], "class_id": 0, "conf": 0.85, "true_distance_m": 8.0},
            {"bbox": [300, 380, 420, 460], "class_id": 1, "conf": 0.80, "true_distance_m": 4.5},
            {"bbox": [620, 370, 680, 440], "class_id": 2, "conf": 0.75, "true_distance_m": 12.0},
        ]
        with open(ground_truth_file, "w") as f:
            json.dump(sample, f, indent=2)
        print(f"  Đã tạo file mẫu: {ground_truth_file}")
        print("  Điền ground truth thực tế vào file rồi chạy lại.\n")

    with open(ground_truth_file) as f:
        data = json.load(f)

    est = DistanceEstimator()
    errors = []

    print(f"\n  {'Class':<12} {'True(m)':>8} {'Pred(m)':>8} {'Err%':>8}")
    print(f"  {'─'*40}")

    for item in data:
        result = est.estimate(
            bbox       = tuple(item["bbox"]),
            class_id   = item["class_id"],
            confidence = item.get("conf", 1.0),
        )
        true_d = item["true_distance_m"]
        err_pct = abs(result.distance_m - true_d) / true_d * 100
        errors.append(err_pct)
        ok = "✅" if err_pct < 20 else "⚠"
        print(f"  {result.class_name:<12} {true_d:>8.1f} {result.distance_m:>8.1f} {err_pct:>7.1f}%  {ok}")

    print(f"  {'─'*40}")
    print(f"  Mean Abs Error: {np.mean(errors):.1f}%  |  Max: {np.max(errors):.1f}%")
    if np.mean(errors) < 20:
        print("  ✅ Đạt mục tiêu sai số < 20%")
    else:
        print("  ⚠  Sai số cao — cần hiệu chỉnh lại camera (--calibrate)")


# ══════════════════════════════════════════════════════════
#  TEST LIVE VỚI VIDEO / WEBCAM
# ══════════════════════════════════════════════════════════
def test_live(source="0"):
    """
    Test ước lượng khoảng cách live (không cần YOLO, dùng click chuột).
    Click vào vật thể để tạo bbox giả và xem khoảng cách ước lượng.
    """
    try:
        src = int(source)
    except ValueError:
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  Không mở được nguồn: {source}")
        return

    est = DistanceEstimator()
    start_pt = None
    current_results = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal start_pt, current_results
        if event == cv2.EVENT_LBUTTONDOWN:
            start_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and start_pt:
            x1, y1 = min(start_pt[0], x), min(start_pt[1], y)
            x2, y2 = max(start_pt[0], x), max(start_pt[1], y)
            if x2 - x1 > 5 and y2 - y1 > 5:
                r = est.estimate(bbox=(x1, y1, x2, y2), class_id=0, confidence=0.9)
                current_results = [r]
            start_pt = None

    cv2.namedWindow("Distance Test — Vẽ bbox bằng chuột")
    cv2.setMouseCallback("Distance Test — Vẽ bbox bằng chuột", mouse_callback)

    print("  Hướng dẫn: Vẽ bbox quanh xe ô tô bằng chuột trái")
    print("  Nhấn Q để thoát\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        if current_results:
            frame = est.draw_overlay(frame, current_results)
            r = current_results[0]
            cv2.putText(frame, f"Dist: {r.distance_m:.1f}m  (focal={est.focal_length_px:.0f}px)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Distance Test — Vẽ bbox bằng chuột", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ước lượng khoảng cách")
    parser.add_argument("--calibrate",  action="store_true", help="Hiệu chỉnh tiêu cự camera")
    parser.add_argument("--benchmark",  action="store_true", help="Đo sai số ước lượng")
    parser.add_argument("--test",       metavar="SOURCE", nargs="?", const="0",
                        help="Test live với webcam (0) hoặc video file")
    parser.add_argument("--gt-file",    default="data/distance_groundtruth.json",
                        help="File ground truth cho benchmark")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  ƯỚC LƯỢNG KHOẢNG CÁCH")
    print(f"{'='*55}")

    if args.calibrate:
        interactive_calibration()
    elif args.benchmark:
        benchmark_accuracy(args.gt_file)
    elif args.test is not None:
        test_live(args.test)
    else:
        # Demo nhanh không cần video
        print("\n  Demo tính toán (không cần camera):\n")
        est = DistanceEstimator(focal_length_px=700.0)
        test_cases = [
            {"bbox": (300, 300, 552, 430), "class_id": 0, "conf": 0.92},  # car gần
            {"bbox": (500, 340, 640, 430), "class_id": 0, "conf": 0.85},  # car xa hơn
            {"bbox": (400, 350, 470, 430), "class_id": 1, "conf": 0.80},  # motorbike
            {"bbox": (610, 360, 650, 430), "class_id": 2, "conf": 0.75},  # person xa
        ]
        results = est.estimate_batch(test_cases)
        print(f"  {'Class':<15} {'BBox W(px)':>12} {'Dist(m)':>10} {'Reliable':>10}")
        print(f"  {'─'*50}")
        for r in results:
            print(f"  {r.class_name:<15} {r.pixel_width:>12.0f} {r.distance_m:>10.1f} {'✅' if r.is_reliable else '⚠':>10}")

        closest = est.get_closest(results)
        if closest:
            print(f"\n  Vật gần nhất: {closest.class_name} @ {closest.distance_m:.1f}m")

        print(f"\n  Cách dùng thực tế:")
        print(f"    python src/distance/estimator.py --calibrate")
        print(f"    python src/distance/estimator.py --test \"demo/videos/video1.mp4\"")
        print(f"    python src/distance/estimator.py --benchmark")

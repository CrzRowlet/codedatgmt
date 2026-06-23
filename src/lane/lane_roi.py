"""
=============================================================
MODULE: Lane ROI Detection & Vehicle Lane Classification
src/lane/lane_roi.py
=============================================================
Chức năng:
  - Định nghĩa ROI (Region of Interest) cho từng làn đường
  - Vẽ các lane polygon lên frame với màu sắc theo mức cảnh báo
  - Phân loại xe thuộc làn nào dựa trên vị trí bounding box
  - Chế độ thiết lập ROI tương tác bằng chuột
  - Hỗ trợ 3 cấu hình: 1 làn / 2 làn / 3 làn

Cách dùng standalone:
  python src/lane/lane_roi.py --setup 0          # Vẽ ROI bằng chuột (webcam)
  python src/lane/lane_roi.py --setup video.mp4  # Vẽ ROI trên video
  python src/lane/lane_roi.py --demo             # Demo với ROI mặc định
  python src/lane/lane_roi.py --test video.mp4   # Test phân loại lane

LÝ DO CẦN ROI LANE:
  Thực tế khi xe chạy, camera hành trình chỉ thấy một phần của đường.
  Xe trong cùng lane mới thực sự là mối nguy hiểm cần cảnh báo.
  Xe ở lane bên cạnh có thể bỏ qua hoặc giảm mức ưu tiên cảnh báo.
"""

import cv2
import numpy as np
import json
import os
import time
import argparse
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum


# ══════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════

# File lưu cấu hình ROI
LANE_CONFIG_FILE = "configs/lane_roi.json"

# Màu sắc BGR cho từng lane
LANE_COLORS = {
    0: (0,   255, 100),   # Lane trái   — xanh lá
    1: (0,   200, 255),   # Lane giữa  — vàng cyan (EGO lane — quan trọng nhất)
    2: (255, 150, 0),     # Lane phải  — cam
    3: (200, 100, 255),   # Lane thêm  — tím
}

# Màu khi xe nằm trong lane
LANE_OCCUPIED_COLOR = {
    0: (50,  255, 150),
    1: (0,   100, 255),   # Đỏ cam khi có xe ở EGO lane
    2: (0,   150, 255),
    3: (150, 100, 255),
}

# Tên lane
LANE_NAMES = {
    0: "LANE TRAI",
    1: "LANE EGO",     # Lane đang chạy — cảnh báo cao nhất
    2: "LANE PHAI",
    3: "LANE THEM",
}

# Mức ưu tiên cảnh báo theo lane (1 = cao nhất)
LANE_PRIORITY = {
    0: 2,   # Lane trái  — ưu tiên 2
    1: 1,   # EGO lane   — ưu tiên cao nhất
    2: 2,   # Lane phải  — ưu tiên 2
    3: 3,   # Lane thêm  — ưu tiên thấp
}


# ══════════════════════════════════════════════════════════
#  ENUM LANE POSITION
# ══════════════════════════════════════════════════════════
class LanePosition(Enum):
    """Vị trí của xe so với các làn đường."""
    UNKNOWN      = "UNKNOWN"
    EGO_LANE     = "EGO_LANE"       # Cùng làn — nguy hiểm nhất
    LEFT_LANE    = "LEFT_LANE"      # Làn trái
    RIGHT_LANE   = "RIGHT_LANE"     # Làn phải
    OUTSIDE_ROI  = "OUTSIDE_ROI"    # Ngoài vùng quan sát

# Ánh xạ LanePosition → mức độ nguy hiểm (hệ số nhân)
LANE_DANGER_MULTIPLIER = {
    LanePosition.EGO_LANE    : 1.0,    # Giữ nguyên mức cảnh báo
    LanePosition.LEFT_LANE   : 0.6,    # Giảm 40%
    LanePosition.RIGHT_LANE  : 0.6,
    LanePosition.OUTSIDE_ROI : 0.2,    # Giảm 80%
    LanePosition.UNKNOWN     : 0.8,
}


# ══════════════════════════════════════════════════════════
#  DATACLASS: Định nghĩa một làn
# ══════════════════════════════════════════════════════════
@dataclass
class Lane:
    """Một làn đường được định nghĩa bởi polygon điểm."""
    lane_id   : int
    name      : str
    points    : List[Tuple[int, int]]   # Danh sách điểm tạo nên polygon
    is_ego    : bool = False            # True nếu đây là làn đang chạy

    @property
    def polygon(self) -> np.ndarray:
        """Trả về polygon dạng numpy array."""
        return np.array(self.points, dtype=np.int32)

    @property
    def color(self) -> Tuple[int, int, int]:
        return LANE_COLORS.get(self.lane_id, (200, 200, 200))

    def contains_point(self, x: int, y: int) -> bool:
        """Kiểm tra điểm (x, y) có nằm trong lane không."""
        if len(self.points) < 3:
            return False
        result = cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False)
        return result >= 0

    def contains_bbox(self, bbox: Tuple[int, int, int, int], threshold: float = 0.4) -> bool:
        """
        Kiểm tra bounding box có nằm trong lane không.
        threshold: Tỷ lệ diện tích bbox phải nằm trong polygon mới tính là 'trong lane'.
        """
        if len(self.points) < 3:
            return False

        x1, y1, x2, y2 = bbox
        # Kiểm tra điểm giữa dưới của bbox (chân của xe)
        cx  = (x1 + x2) // 2
        cy  = y2   # Dùng cạnh dưới — điểm tiếp đất

        # Kiểm tra điểm giữa-dưới và hai góc dưới
        check_points = [
            (cx, cy),                       # Giữa-dưới
            (x1 + (x2 - x1) // 4, cy),      # 1/4 trái-dưới
            (x2 - (x2 - x1) // 4, cy),      # 1/4 phải-dưới
        ]

        hits = sum(1 for px, py in check_points
                   if cv2.pointPolygonTest(self.polygon, (float(px), float(py)), False) >= 0)

        return hits >= 2   # Ít nhất 2/3 điểm phải trong polygon

    def to_dict(self) -> dict:
        return {
            "lane_id": self.lane_id,
            "name"   : self.name,
            "points" : self.points,
            "is_ego" : self.is_ego,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Lane":
        return cls(
            lane_id = data["lane_id"],
            name    = data["name"],
            points  = [tuple(p) for p in data["points"]],
            is_ego  = data.get("is_ego", False),
        )


# ══════════════════════════════════════════════════════════
#  CLASS CHÍNH: LaneROI
# ══════════════════════════════════════════════════════════
class LaneROI:
    """
    Quản lý ROI làn đường và phân loại xe theo làn.

    Workflow:
      1. Tạo LaneROI()
      2. Gọi setup_interactive() để vẽ ROI bằng chuột (lần đầu)
         hoặc load_config() nếu đã có cấu hình
      3. Trong pipeline: gọi classify_detection() cho mỗi xe
      4. Gọi draw_lanes() để vẽ ROI lên frame

    Cấu trúc ROI đề xuất cho camera hành trình xe hơi:
      Frame 1280x720:
        ┌─────────────────────────────────┐
        │          (horizon line)         │  ← y ≈ 300-380 (40-50% height)
        │    ╱▔▔▔▔▔▔▔╲ ╱▔▔▔▔▔▔▔╲         │
        │   ╱  LEFT   ╲╱   EGO  ╲        │
        │  ╱____________╲___RIGHT╲       │
        │     (bottom)     (bottom)       │  ← y ≈ 680-720
        └─────────────────────────────────┘
    """

    def __init__(
        self,
        frame_width : int = 1280,
        frame_height: int = 720,
        config_file : str = LANE_CONFIG_FILE,
        n_lanes     : int = 3,   # Số làn: 1, 2, hoặc 3
    ):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.config_file  = config_file
        self.n_lanes      = n_lanes
        self.lanes        : List[Lane] = []
        self._alpha       = 0.25    # Độ trong suốt overlay polygon
        self._is_setup    = False   # Đã thiết lập ROI chưa

        # Thử load config
        loaded = self._load_config()
        if not loaded:
            # Tạo ROI mặc định
            self._create_default_lanes()

    # ── Load / Save config ────────────────────────────────
    def _load_config(self) -> bool:
        """Load cấu hình ROI từ file. Trả về True nếu thành công."""
        if not os.path.exists(self.config_file):
            return False
        try:
            with open(self.config_file) as f:
                data = json.load(f)
            self.lanes = [Lane.from_dict(d) for d in data.get("lanes", [])]
            self.frame_width  = data.get("frame_width",  self.frame_width)
            self.frame_height = data.get("frame_height", self.frame_height)
            self._is_setup = True
            print(f"[LaneROI] Đã load {len(self.lanes)} làn từ: {self.config_file}")
            return True
        except Exception as e:
            print(f"[LaneROI] Lỗi load config: {e}")
            return False

    def save_config(self):
        """Lưu cấu hình ROI ra file."""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        data = {
            "frame_width" : self.frame_width,
            "frame_height": self.frame_height,
            "lanes"       : [lane.to_dict() for lane in self.lanes],
            "saved_at"    : time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[LaneROI] Đã lưu {len(self.lanes)} làn vào: {self.config_file}")

    def update_frame_size(self, width: int, height: int):
        """Cập nhật kích thước frame và scale lại ROI nếu khác."""
        if width == self.frame_width and height == self.frame_height:
            return
        sx = width  / max(self.frame_width,  1)
        sy = height / max(self.frame_height, 1)
        for lane in self.lanes:
            lane.points = [(int(x * sx), int(y * sy)) for x, y in lane.points]
        self.frame_width  = width
        self.frame_height = height
        print(f"[LaneROI] Scale ROI từ {width}x{height} → scale ({sx:.2f}, {sy:.2f})")

    # ── Tạo ROI mặc định ─────────────────────────────────
    def _create_default_lanes(self):
        """
        Tạo ROI mặc định cho camera hành trình.
        Hình thang thu hẹp về phía horizon.
        """
        w, h = self.frame_width, self.frame_height
        # Horizon line hạ xuống 65% chiều cao (sát mặt đường hơn)
        hy = int(h * 0.65)
        # Bottom y ≈ 97% chiều cao
        by = int(h * 0.97)

        if self.n_lanes == 1:
            # Chỉ 1 làn trung tâm (EGO)
            self.lanes = [
                Lane(lane_id=1, name="LANE EGO", is_ego=True, points=[
                    (int(w * 0.40), hy),
                    (int(w * 0.60), hy),
                    (int(w * 0.85), by),
                    (int(w * 0.15), by),
                ]),
            ]

        elif self.n_lanes == 2:
            # 2 làn: trái + phải (EGO)
            mid_top = int(w * 0.50)
            mid_bot = int(w * 0.50)
            self.lanes = [
                Lane(lane_id=0, name="LANE TRAI", is_ego=False, points=[
                    (int(w * 0.15), hy),
                    (mid_top,       hy),
                    (mid_bot,       by),
                    (int(w * -0.30), by),
                ]),
                Lane(lane_id=1, name="LANE EGO", is_ego=True, points=[
                    (mid_top,       hy),
                    (int(w * 0.85), hy),
                    (int(w * 1.30), by),
                    (mid_bot,       by),
                ]),
            ]

        else:  # 3 làn (mặc định)
            # 3 làn: trái / EGO (giữa) / phải
            left_top  = int(w * 0.15)
            right_top = int(w * 0.85)
            left_bot  = int(w * -0.50)
            right_bot = int(w * 1.50)
            mid_left_top = int(w * 0.40)
            mid_right_top = int(w * 0.60)
            mid_left_bot  = int(w * 0.15)
            mid_right_bot = int(w * 0.85)

            self.lanes = [
                Lane(lane_id=0, name="LANE TRAI", is_ego=False, points=[
                    (left_top,      hy),
                    (mid_left_top,  hy),
                    (mid_left_bot,  by),
                    (left_bot,      by),
                ]),
                Lane(lane_id=1, name="LANE EGO", is_ego=True, points=[
                    (mid_left_top,  hy),
                    (mid_right_top, hy),
                    (mid_right_bot, by),
                    (mid_left_bot,  by),
                ]),
                Lane(lane_id=2, name="LANE PHAI", is_ego=False, points=[
                    (mid_right_top, hy),
                    (right_top,     hy),
                    (right_bot,     by),
                    (mid_right_bot, by),
                ]),
            ]

        self._is_setup = True
        print(f"[LaneROI] Tạo ROI mặc định: {self.n_lanes} làn ({self.frame_width}x{self.frame_height})")

    # ── Phân loại xe theo làn ─────────────────────────────
    def classify_bbox(self, bbox: Tuple[int, int, int, int]) -> LanePosition:
        """
        Xác định xe thuộc làn nào dựa trên bounding box.

        Args:
            bbox: (x1, y1, x2, y2)

        Returns:
            LanePosition enum
        """
        if not self.lanes:
            return LanePosition.UNKNOWN

        for lane in self.lanes:
            if lane.contains_bbox(bbox):
                if lane.is_ego:
                    return LanePosition.EGO_LANE
                elif lane.lane_id == 0:
                    return LanePosition.LEFT_LANE
                elif lane.lane_id == 2:
                    return LanePosition.RIGHT_LANE
                else:
                    return LanePosition.EGO_LANE  # Nếu chỉ có 1 làn hoặc thêm

        return LanePosition.OUTSIDE_ROI

    def classify_detection(self, det: dict) -> dict:
        """
        Thêm thông tin lane_position vào dict detection/track.

        Args:
            det: dict với key 'bbox' = (x1, y1, x2, y2)

        Returns:
            dict mới với 'lane_position' và 'lane_priority' được thêm vào
        """
        bbox = det.get("bbox") or det.get("last_bbox")
        if bbox is None:
            return {**det, "lane_position": LanePosition.UNKNOWN, "lane_priority": 3}

        pos = self.classify_bbox(tuple(bbox))
        priority = LANE_PRIORITY.get(
            {LanePosition.EGO_LANE: 1,
             LanePosition.LEFT_LANE: 0,
             LanePosition.RIGHT_LANE: 2,
             LanePosition.OUTSIDE_ROI: 3}.get(pos, 3),
            3
        )
        return {
            **det,
            "lane_position" : pos,
            "lane_priority" : priority,
            "lane_multiplier": LANE_DANGER_MULTIPLIER.get(pos, 0.8),
        }

    def classify_tracks(self, tracks: List[dict]) -> List[dict]:
        """
        Thêm lane_position vào tất cả tracks.
        Sắp xếp lại: EGO lane trước, sau đó theo khoảng cách.
        """
        classified = [self.classify_detection(t) for t in tracks]
        # Sắp xếp: EGO lane trước, rồi theo khoảng cách
        classified.sort(key=lambda t: (
            0 if t["lane_position"] == LanePosition.EGO_LANE else 1,
            t.get("distance_m", 99),
        ))
        return classified

    # ── Vẽ ROI lên frame ──────────────────────────────────
    def draw(
        self,
        frame       : np.ndarray,
        tracks      : List[dict] = None,
        show_labels : bool = True,
        alpha       : float = None,
    ) -> np.ndarray:
        """
        Vẽ tất cả lane ROI lên frame.

        Args:
            frame      : BGR frame gốc
            tracks     : List track (để tô màu lane đang có xe)
            show_labels: Hiển thị nhãn tên làn
            alpha      : Độ trong suốt (0=trong suốt, 1=đục)

        Returns:
            Frame đã vẽ ROI
        """
        if not self.lanes:
            return frame

        alpha = alpha or self._alpha
        overlay = frame.copy()

        # Xác định lane nào đang có xe
        occupied_lanes = set()
        if tracks:
            for t in tracks:
                pos = t.get("lane_position")
                if pos == LanePosition.EGO_LANE:
                    # Tìm lane có is_ego=True
                    for lane in self.lanes:
                        if lane.is_ego:
                            occupied_lanes.add(lane.lane_id)
                elif pos == LanePosition.LEFT_LANE:
                    occupied_lanes.add(0)
                elif pos == LanePosition.RIGHT_LANE:
                    occupied_lanes.add(2)

        for lane in self.lanes:
            pts = lane.polygon.reshape((-1, 1, 2))
            is_occupied = lane.lane_id in occupied_lanes

            # Chọn màu: nếu có xe thì sáng hơn
            if is_occupied and lane.is_ego:
                fill_color = (0, 60, 180)    # Đỏ cam — EGO lane có xe
                border_color = (0, 100, 255)
                border_thick = 3
            elif is_occupied:
                fill_color   = LANE_OCCUPIED_COLOR.get(lane.lane_id, (100, 200, 100))
                border_color = fill_color
                border_thick = 2
            else:
                fill_color   = lane.color
                border_color = tuple(min(255, c + 60) for c in lane.color)
                border_thick = 1

            # Vẽ fill mờ
            cv2.fillPoly(overlay, [pts], fill_color)
            # Vẽ viền
            cv2.polylines(frame, [pts], isClosed=True, color=border_color, thickness=border_thick)

        # Blend overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Vẽ nhãn làn
        if show_labels:
            for lane in self.lanes:
                self._draw_lane_label(frame, lane, lane.lane_id in occupied_lanes)

        return frame

    def _draw_lane_label(self, frame: np.ndarray, lane: Lane, is_occupied: bool):
        """Vẽ nhãn tên làn ở vị trí trung tâm polygon."""
        if len(lane.points) < 3:
            return

        pts = np.array(lane.points)
        # Vị trí nhãn: trung tâm của nửa dưới polygon
        lower_pts = [p for p in lane.points if p[1] > self.frame_height * 0.6]
        if not lower_pts:
            lower_pts = lane.points

        cx = int(np.mean([p[0] for p in lower_pts]))
        cy = int(np.mean([p[1] for p in lower_pts]))

        label = lane.name
        color = lane.color if not is_occupied else (0, 80, 255)

        if lane.is_ego and is_occupied:
            label = f"! {lane.name} !"

        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.50
        thick = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

        # Nền nhãn
        cv2.rectangle(frame, (cx - tw//2 - 3, cy - th - 4),
                      (cx + tw//2 + 3, cy + 4), (20, 20, 20), -1)
        cv2.putText(frame, label, (cx - tw//2, cy),
                    font, scale, color, thick, cv2.LINE_AA)

    # ── Vẽ điểm footprint của xe ──────────────────────────
    def draw_vehicle_footprints(self, frame: np.ndarray, tracks: List[dict]) -> np.ndarray:
        """Vẽ điểm chân (footprint) của từng xe lên ROI để debug."""
        for t in tracks:
            bbox = t.get("bbox") or t.get("last_bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            pos = t.get("lane_position", LanePosition.UNKNOWN)

            color_map = {
                LanePosition.EGO_LANE   : (0, 0, 255),
                LanePosition.LEFT_LANE  : (50, 200, 50),
                LanePosition.RIGHT_LANE : (0, 165, 255),
                LanePosition.OUTSIDE_ROI: (100, 100, 100),
                LanePosition.UNKNOWN    : (180, 180, 180),
            }
            color = color_map.get(pos, (180, 180, 180))
            cv2.circle(frame, (cx, y2), 5, color, -1)
            cv2.circle(frame, (cx, y2), 7, (255, 255, 255), 1)

        return frame

    # ══════════════════════════════════════════════════════
    #  THIẾT LẬP ROI TƯƠNG TÁC BẰNG CHUỘT
    # ══════════════════════════════════════════════════════
    def setup_interactive(self, source=0) -> bool:
        """
        Thiết lập ROI bằng chuột.
        Người dùng click để đặt các điểm polygon cho từng làn.

        Controls:
          Click trái : Thêm điểm vào làn hiện tại
          SPACE      : Hoàn thành làn hiện tại, sang làn tiếp theo
          R          : Reset làn hiện tại
          S          : Lưu và thoát
          Q / ESC    : Thoát không lưu

        Returns:
            True nếu đã lưu, False nếu thoát không lưu
        """
        try:
            src = int(source)
        except (ValueError, TypeError):
            src = source

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[LaneROI] Không mở được nguồn: {source}")
            return False

        ret, base_frame = cap.read()
        if not ret:
            cap.release()
            print("[LaneROI] Không đọc được frame đầu tiên!")
            return False

        self.frame_height, self.frame_width = base_frame.shape[:2]
        cap.release()

        # State cho mouse callback
        state = {
            "current_lane_idx": 0,
            "current_points"  : [],
            "all_lanes"       : [],
            "done"            : False,
        }

        n_lanes = self.n_lanes
        lane_names = [LANE_NAMES.get(i, f"LANE {i}") for i in range(n_lanes)]

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if state["current_lane_idx"] < n_lanes:
                    state["current_points"].append((x, y))

        win = "Setup Lane ROI"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.setMouseCallback(win, mouse_cb)

        print("\n" + "="*55)
        print("  THIẾT LẬP ROI LÀN ĐƯỜNG")
        print("="*55)
        print(f"  Số làn cần vẽ: {n_lanes}")
        print("  Click trái: Đặt điểm")
        print("  SPACE     : Xong làn này → sang làn tiếp theo")
        print("  R         : Reset làn hiện tại")
        print("  S         : Lưu tất cả ROI")
        print("  Q / ESC   : Thoát không lưu")
        print("="*55)
        print(f"\n  → Đang vẽ: {lane_names[0]}")
        print("    Khuyến nghị: Vẽ hình thang thu hẹp về phía horizon (top)")

        while True:
            frame = base_frame.copy()
            ci = state["current_lane_idx"]

            # Vẽ các làn đã hoàn thành
            for lane in state["all_lanes"]:
                pts = np.array(lane.points, dtype=np.int32).reshape((-1, 1, 2))
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], lane.color)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                cv2.polylines(frame, [pts], True, lane.color, 2)

            # Vẽ làn đang vẽ
            pts_now = state["current_points"]
            if len(pts_now) > 0:
                color_now = LANE_COLORS.get(ci, (200, 200, 200))
                for p in pts_now:
                    cv2.circle(frame, p, 5, color_now, -1)
                    cv2.circle(frame, p, 7, (255, 255, 255), 1)
                if len(pts_now) > 1:
                    cv2.polylines(frame, [np.array(pts_now, dtype=np.int32)],
                                  False, color_now, 2)
                if len(pts_now) > 2:
                    pts_arr = np.array(pts_now, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts_arr], True, color_now, 1)

            # Hướng dẫn
            guide_texts = []
            if ci < n_lanes:
                guide_texts.append(f"Dang ve: {lane_names[ci]}  ({len(pts_now)} diem)")
                guide_texts.append(f"SPACE: Xong lam nay  |  R: Reset  |  S: Luu  |  Q: Thoat")
            else:
                guide_texts.append("Tat ca lan da ve xong!  Nhan S de luu.")

            for i, txt in enumerate(guide_texts):
                cv2.putText(frame, txt, (10, 25 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

            # Progress
            cv2.putText(frame,
                        f"Lane {min(ci, n_lanes-1)+1}/{n_lanes}: {lane_names[min(ci,n_lanes-1)]}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord(' '):   # SPACE — Hoàn thành làn hiện tại
                if len(pts_now) >= 3 and ci < n_lanes:
                    is_ego = (ci == 1) if n_lanes > 1 else True
                    lane = Lane(
                        lane_id = ci,
                        name    = lane_names[ci],
                        points  = pts_now.copy(),
                        is_ego  = is_ego,
                    )
                    state["all_lanes"].append(lane)
                    state["current_points"] = []
                    state["current_lane_idx"] += 1
                    if ci + 1 < n_lanes:
                        print(f"\n  ✓ Hoàn thành {lane_names[ci]}")
                        print(f"  → Đang vẽ: {lane_names[ci+1]}")
                    else:
                        print(f"\n  ✓ Hoàn thành {lane_names[ci]}")
                        print("  Tất cả làn xong! Nhấn S để lưu.")
                else:
                    print("  ⚠ Cần ít nhất 3 điểm để tạo polygon!")

            elif key == ord('r') or key == ord('R'):   # Reset
                state["current_points"] = []
                print(f"  Reset làn {lane_names[min(ci, n_lanes-1)]}")

            elif key == ord('s') or key == ord('S'):   # Lưu
                if state["all_lanes"]:
                    self.lanes = state["all_lanes"]
                    self._is_setup = True
                    self.save_config()
                    print(f"\n  ✅ Đã lưu {len(self.lanes)} làn!")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("  ⚠ Chưa có làn nào được vẽ!")

            elif key == ord('q') or key == 27:   # Quit
                print("  Thoát không lưu.")
                cv2.destroyAllWindows()
                return False

        cv2.destroyAllWindows()
        return False

    # ── Thống kê làn ──────────────────────────────────────
    def get_lane_stats(self, tracks: List[dict]) -> Dict[str, int]:
        """Đếm số xe trong mỗi làn."""
        stats = {lane.name: 0 for lane in self.lanes}
        stats["OUTSIDE_ROI"] = 0
        for t in tracks:
            pos = t.get("lane_position", LanePosition.UNKNOWN)
            if pos == LanePosition.EGO_LANE:
                for lane in self.lanes:
                    if lane.is_ego:
                        stats[lane.name] = stats.get(lane.name, 0) + 1
            elif pos == LanePosition.LEFT_LANE:
                stats[LANE_NAMES.get(0, "LANE TRAI")] = stats.get(LANE_NAMES.get(0, "LANE TRAI"), 0) + 1
            elif pos == LanePosition.RIGHT_LANE:
                stats[LANE_NAMES.get(2, "LANE PHAI")] = stats.get(LANE_NAMES.get(2, "LANE PHAI"), 0) + 1
            else:
                stats["OUTSIDE_ROI"] = stats.get("OUTSIDE_ROI", 0) + 1
        return stats


# ══════════════════════════════════════════════════════════
#  HELPER FUNCTION — dùng trong pipeline
# ══════════════════════════════════════════════════════════
def draw_lane_overlay(
    frame  : np.ndarray,
    lane_roi: LaneROI,
    tracks : List[dict] = None,
) -> np.ndarray:
    """
    Hàm tiện ích: vẽ ROI làn + footprint xe lên frame.
    Gọi trong _process_frame() sau khi vẽ bbox.
    """
    if lane_roi is None or not lane_roi.lanes:
        return frame
    lane_roi.draw(frame, tracks=tracks, show_labels=True)
    if tracks:
        lane_roi.draw_vehicle_footprints(frame, tracks)
    return frame


# ══════════════════════════════════════════════════════════
#  DEMO & TEST
# ══════════════════════════════════════════════════════════
def demo_static():
    """Demo với ảnh tĩnh (không cần camera)."""
    print("\n  DEMO STATIC — LaneROI\n")
    w, h = 1280, 720
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Vẽ đường giả
    cv2.line(frame, (0, int(h * 0.45)), (w, int(h * 0.45)), (40, 40, 40), 1)
    for i in range(0, w, 60):
        cv2.line(frame, (i, int(h * 0.70)), (i + 30, int(h * 0.70)), (60, 60, 60), 2)

    roi = LaneROI(frame_width=w, frame_height=h, n_lanes=3)

    # Giả lập một số xe
    fake_tracks = [
        {"bbox": (580, 580, 700, 660), "distance_m": 5.0, "class_name": "car"},
        {"bbox": (800, 560, 920, 640), "distance_m": 8.0, "class_name": "car"},
        {"bbox": (300, 570, 430, 650), "distance_m": 7.0, "class_name": "motorbike"},
    ]
    classified = roi.classify_tracks(fake_tracks)
    for t in classified:
        print(f"  {t['class_name']:12s} @ {t['distance_m']:.1f}m → {t.get('lane_position').value}")

    draw_lane_overlay(frame, roi, classified)

    # Vẽ bbox giả
    for t in classified:
        x1, y1, x2, y2 = t["bbox"]
        pos = t.get("lane_position")
        color_map = {
            LanePosition.EGO_LANE   : (0, 0, 255),
            LanePosition.LEFT_LANE  : (50, 220, 50),
            LanePosition.RIGHT_LANE : (0, 165, 255),
            LanePosition.OUTSIDE_ROI: (100, 100, 100),
        }
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_map.get(pos, (200, 200, 200)), 2)

    cv2.imshow("LaneROI Demo", frame)
    print("\n  Nhấn phím bất kỳ để đóng...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_live(source):
    """Test live với video/webcam."""
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  Không mở được: {source}")
        return

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    roi  = LaneROI(frame_width=w, frame_height=h, n_lanes=3)

    print("  Nhấn Q để thoát")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        draw_lane_overlay(frame, roi, tracks=None)
        cv2.imshow("LaneROI Live", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lane ROI Module")
    parser.add_argument("--setup", metavar="SOURCE", nargs="?", const="0",
                        help="Thiết lập ROI tương tác (webcam=0 hoặc video path)")
    parser.add_argument("--demo",  action="store_true", help="Demo static không cần camera")
    parser.add_argument("--test",  metavar="SOURCE", help="Test live với video/webcam")
    parser.add_argument("--lanes", type=int, default=3, help="Số làn đường (1/2/3)")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  MODULE: LANE ROI DETECTION")
    print(f"{'='*55}")

    if args.setup is not None:
        roi = LaneROI(n_lanes=args.lanes)
        roi.setup_interactive(source=args.setup)
    elif args.test:
        test_live(args.test)
    else:
        demo_static()

"""
=============================================================
SCRIPT: Chuẩn bị dataset YOLOv8
Hệ thống cảnh báo va chạm (Collision Warning System)
=============================================================
Cách dùng:
  python scripts/prepare_dataset.py --verify        # Kiểm tra dataset
  python scripts/prepare_dataset.py --visualize 16  # Hiển thị 16 ảnh mẫu có nhãn
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# ── Cấu hình ──────────────────────────────────────────────
CLASS_NAMES  = ["car", "motorbike", "person", "truck"]
CLASS_COLORS = {
    0: (50, 200, 50),     # car       — xanh lá
    1: (0, 165, 255),     # motorbike — cam
    2: (255, 100, 0),     # person    — xanh dương
    3: (200, 0, 200),     # truck     — tím
}
SPLITS_DIR   = "data/splits"

# ─────────────────────────────────────────────────────────
#  KIỂM TRA & XÁC NHẬN DATASET
# ─────────────────────────────────────────────────────────
def verify_dataset() -> bool:
    """
    Kiểm tra toàn vẹn dataset:
    - Số lượng ảnh khớp nhãn
    - Format YOLO hợp lệ
    - Thống kê phân bố class
    """
    print("\n── BƯỚC 1: Xác nhận dataset ──")

    class_counts = {i: {"train": 0, "val": 0, "test": 0} for i in range(len(CLASS_NAMES))}
    all_ok       = True
    errors       = []

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(SPLITS_DIR, split, "images")
        lbl_dir = os.path.join(SPLITS_DIR, split, "labels")

        if not os.path.exists(img_dir):
            print(f"  ⚠  Split '{split}' chưa có dữ liệu.")
            continue

        images = [p for p in Path(img_dir).glob("*") if p.name != ".gitkeep"]
        print(f"\n  {split.upper()} ({len(images)} ảnh):")

        for img_path in images:
            lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
            if not lbl_path.exists():
                errors.append(f"Thiếu nhãn: {img_path.name}")
                all_ok = False
                continue

            # Kiểm tra từng dòng nhãn (YOLO format: class cx cy w h)
            with open(lbl_path) as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    errors.append(f"{lbl_path.name} dòng {i}: cần 5 giá trị, có {len(parts)}")
                    all_ok = False
                    continue

                try:
                    cls_id = int(parts[0])
                    vals   = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(f"{lbl_path.name} dòng {i}: giá trị không hợp lệ")
                    all_ok = False
                    continue

                # Kiểm tra class ID hợp lệ
                if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                    errors.append(f"{lbl_path.name} dòng {i}: class_id={cls_id} không hợp lệ")
                    all_ok = False
                    continue

                # Kiểm tra tọa độ nằm trong [0, 1]
                if not all(0.0 <= v <= 1.0 for v in vals):
                    errors.append(f"{lbl_path.name} dòng {i}: tọa độ ngoài [0,1]")
                    all_ok = False
                    continue

                class_counts[cls_id][split] += 1

    # In phân bố class
    print("\n  PHÂN BỐ CLASS:")
    print(f"  {'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'─'*47}")
    for cls_id, name in enumerate(CLASS_NAMES):
        tr = class_counts[cls_id]["train"]
        va = class_counts[cls_id]["val"]
        te = class_counts[cls_id]["test"]
        total = tr + va + te
        print(f"  {name:<15} {tr:>8} {va:>8} {te:>8} {total:>8}")

    if errors:
        print(f"\n  ⚠  Phát hiện {len(errors)} lỗi:")
        for e in errors[:10]:
            print(f"     • {e}")
        if len(errors) > 10:
            print(f"     ... và {len(errors)-10} lỗi khác")
    else:
        print("\n  ✅ Tất cả nhãn hợp lệ — Dataset sẵn sàng huấn luyện!")

    return all_ok


# ─────────────────────────────────────────────────────────
#  VISUALIZE ẢNH MẪU VỚI BOUNDING BOX
# ─────────────────────────────────────────────────────────
def visualize_samples(n: int = 16, split: str = "train"):
    """Hiển thị n ảnh mẫu với bounding box được vẽ."""
    img_dir = os.path.join(SPLITS_DIR, split, "images")
    lbl_dir = os.path.join(SPLITS_DIR, split, "labels")

    if not os.path.exists(img_dir):
        print(f"  ✗ Chưa có dữ liệu split '{split}'")
        return

    images = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png"))
    if not images:
        print(f"  ✗ Không tìm thấy ảnh trong {img_dir}")
        return

    sample = random.sample(images, min(n, len(images)))

    # Tạo grid ảnh
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    cell_size = (320, 240)
    canvas = np.zeros((rows * cell_size[1], cols * cell_size[0], 3), dtype=np.uint8)

    for idx, img_path in enumerate(sample):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")

        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = [float(x) for x in parts[1:]]
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                    cv2.putText(img, label, (x1, max(y1-5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Resize và đặt vào grid
        img_resized = cv2.resize(img, cell_size)
        r = idx // cols
        c = idx  % cols
        canvas[r*cell_size[1]:(r+1)*cell_size[1],
               c*cell_size[0]:(c+1)*cell_size[0]] = img_resized

    # Lưu ảnh preview
    out_path = "data/splits/preview_grid.jpg"
    cv2.imwrite(out_path, canvas)
    print(f"\n  ✅ Preview grid lưu tại: {out_path}")
    print(f"  Mở file để kiểm tra bounding box trực quan.")

    try:
        cv2.imshow(f"Dataset Preview - {split} ({len(sample)} ảnh)", canvas)
        print("  Nhấn phím bất kỳ để đóng cửa sổ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("  (Không hiển thị được window — xem ảnh tại out_path)")

# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra và xem trước Dataset YOLOv8"
    )
    parser.add_argument("--verify", action="store_true",
        help="Kiểm tra toàn vẹn dataset")
    parser.add_argument("--visualize", type=int, metavar="N", default=0,
        help="Hiển thị N ảnh mẫu với bounding box (mặc định: 16)")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════════╗")
    print("║   GÁN NHÃN & CHUẨN BỊ DATASET                    ║")
    print("╚══════════════════════════════════════════════════╝")

    if args.verify:
        verify_dataset()

    if args.visualize > 0:
        visualize_samples(args.visualize)

    if not any([args.verify, args.visualize]):
        print("\n  Cách dùng:")
        print("    python scripts/prepare_dataset.py --verify     # Kiểm tra nhãn")
        print("    python scripts/prepare_dataset.py --visualize 16  # Xem mẫu")

if __name__ == "__main__":
    main()

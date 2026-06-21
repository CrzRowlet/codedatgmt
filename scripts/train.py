"""
=============================================================
SCRIPT: Huấn luyện YOLOv8
=============================================================
Cách dùng:
  python scripts/train.py --train        # Bắt đầu huấn luyện
  python scripts/train.py --evaluate     # Đánh giá model tốt nhất
  python scripts/train.py --export       # Xuất ONNX
  python scripts/train.py --all          # Chạy toàn bộ pipeline
"""

import os
import sys
import shutil
import argparse
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

# ── Cấu hình ──────────────────────────────────────────────
DATA_YAML        = "configs/data.yaml"
TRAIN_DIR        = "runs/detect/train"
WEIGHTS_DIR      = "models/weights"

# Hyperparameters huấn luyện
TRAIN_CONFIG = {
    "model"   : "yolov8s.pt",    # yolov8n (nhanh) | yolov8s (khuyến nghị) | yolov8m (chính xác hơn)
    "epochs"  : 100,
    "imgsz"   : 640,
    "batch"   : 16,              # Giảm xuống 8 nếu hết VRAM
    "lr0"     : 0.01,
    "lrf"     : 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "patience": 20,              # Early stopping
    "workers" : 4,
    "device"  : "auto",          # auto | cpu | 0 (GPU 0) | mps
    "project" : "runs/detect",
    "name"    : "collision_yolov8",
    "exist_ok": True,
    "cache"   : False,
    "amp"     : True,            # Mixed precision (FP16) — tăng tốc GPU
    "plots"   : True,
    "save"    : True,
    "verbose" : True,
}

# Class names
CLASS_NAMES = ["car", "motorbike", "person", "truck"]

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def log(msg):  print(f"{GREEN}{msg}{RESET}")
def info(msg): print(f"{CYAN}  {msg}{RESET}")
def warn(msg): print(f"{YELLOW}⚠  {msg}{RESET}")
def section(t): print(f"\n{BOLD}{CYAN}{'='*55}\n  {t}\n{'='*55}{RESET}")


# ══════════════════════════════════════════════════════════
#  BƯỚC 1: HUẤN LUYỆN YOLOV8
# ══════════════════════════════════════════════════════════
def train_model():
    section("BƯỚC 1 — Huấn luyện YOLOv8")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  Chưa cài Ultralytics: pip install ultralytics")
        sys.exit(1)

    if not Path(DATA_YAML).exists():
        warn(f"Chưa có {DATA_YAML}. Hãy đảm bảo đã cấu hình đúng file YAML.")
        return None

    info(f"Backbone     : {TRAIN_CONFIG['model']}")
    info(f"Dataset      : {DATA_YAML}")
    info(f"Epochs       : {TRAIN_CONFIG['epochs']}")
    info(f"Image size   : {TRAIN_CONFIG['imgsz']}px")
    info(f"Batch size   : {TRAIN_CONFIG['batch']}")
    print()

    # Kiểm tra device
    import torch
    if torch.cuda.is_available():
        device_info = torch.cuda.get_device_name(0)
        info(f"Device       : CUDA — {device_info}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info("Device       : Apple MPS")
    else:
        warn("Device       : CPU (chậm — khuyến nghị dùng Google Colab GPU)")

    print(f"\n  Bắt đầu huấn luyện lúc: {datetime.now().strftime('%H:%M:%S')}")
    print("  " + "─"*50)

    model = YOLO(TRAIN_CONFIG["model"])
    results = model.train(
        data      = DATA_YAML,
        epochs    = TRAIN_CONFIG["epochs"],
        imgsz     = TRAIN_CONFIG["imgsz"],
        batch     = TRAIN_CONFIG["batch"],
        lr0       = TRAIN_CONFIG["lr0"],
        lrf       = TRAIN_CONFIG["lrf"],
        momentum  = TRAIN_CONFIG["momentum"],
        weight_decay = TRAIN_CONFIG["weight_decay"],
        warmup_epochs = TRAIN_CONFIG["warmup_epochs"],
        patience  = TRAIN_CONFIG["patience"],
        workers   = TRAIN_CONFIG["workers"],
        device    = TRAIN_CONFIG["device"],
        project   = TRAIN_CONFIG["project"],
        name      = TRAIN_CONFIG["name"],
        exist_ok  = TRAIN_CONFIG["exist_ok"],
        cache     = TRAIN_CONFIG["cache"],
        amp       = TRAIN_CONFIG["amp"],
        plots     = TRAIN_CONFIG["plots"],
        save      = TRAIN_CONFIG["save"],
        verbose   = TRAIN_CONFIG["verbose"],
    )

    # Copy best.pt vào thư mục models/
    best_pt = Path(TRAIN_CONFIG["project"]) / TRAIN_CONFIG["name"] / "weights" / "best.pt"
    
    # Tìm kiếm thêm nếu YOLO lưu ở thư mục nested khác do thiết lập mặc định của Windows/Ultralytics
    if not best_pt.exists():
        candidates = list(Path("runs").rglob("best.pt")) if Path("runs").exists() else []
        if candidates:
            # Sắp xếp theo thời gian sửa đổi mới nhất
            candidates.sort(key=lambda p: p.stat().st_mtime)
            best_pt = candidates[-1]

    if best_pt.exists():
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        dst = Path(WEIGHTS_DIR) / "best.pt"
        shutil.copy2(best_pt, dst)
        log(f"\n  ✅ best.pt đã lưu tại: {dst}")
    else:
        warn("\n  ✗ Không tìm thấy file best.pt sau khi train.")

    return results


# ══════════════════════════════════════════════════════════
#  BƯỚC 2: ĐÁNH GIÁ MODEL
# ══════════════════════════════════════════════════════════
def evaluate_model():
    section("BƯỚC 2 — Đánh giá model")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  pip install ultralytics")
        return

    best_pt = Path(WEIGHTS_DIR) / "best.pt"
    if not best_pt.exists():
        # Tìm best.pt trong runs/
        candidates = list(Path("runs").rglob("best.pt")) if Path("runs").exists() else []
        if not candidates:
            warn(f"Chưa có best.pt. Chạy --train trước.")
            return
        # Sắp xếp theo thời gian sửa đổi mới nhất
        candidates.sort(key=lambda p: p.stat().st_mtime)
        found_pt = candidates[-1]
        info(f"Tìm thấy best.pt tại: {found_pt}. Đang copy vào {best_pt}...")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        shutil.copy2(found_pt, best_pt)

    model   = YOLO(str(best_pt))
    metrics = model.val(data=DATA_YAML, imgsz=640, batch=16, verbose=True)

    # Lấy kết quả
    map50    = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall    = float(metrics.box.mr)

    print(f"\n  {'─'*45}")
    print(f"  {'Metric':<25} {'Giá trị':>10}")
    print(f"  {'─'*45}")
    print(f"  {'mAP@0.5':<25} {map50:>10.4f}  {'✅' if map50>=0.70 else '⚠ cần cải thiện'}")
    print(f"  {'mAP@0.5:0.95':<25} {map50_95:>10.4f}")
    print(f"  {'Precision':<25} {precision:>10.4f}")
    print(f"  {'Recall':<25} {recall:>10.4f}")
    print(f"  {'─'*45}")

    # Đánh giá theo class
    if hasattr(metrics.box, "ap_class_index"):
        print(f"\n  mAP@0.5 theo class:")
        for i, ap in enumerate(metrics.box.maps):
            name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}"
            bar  = "█" * int(ap * 20)
            print(f"    {name:<15} {ap:.4f}  {bar}")

    # Lưu kết quả vào JSON
    eval_result = {
        "timestamp" : datetime.now().isoformat(),
        "model"     : str(best_pt),
        "mAP50"     : round(map50, 4),
        "mAP50_95"  : round(map50_95, 4),
        "precision" : round(precision, 4),
        "recall"    : round(recall, 4),
        "target_met": map50 >= 0.70,
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_results.json", "w") as f:
        json.dump(eval_result, f, indent=2)
    info("Kết quả lưu tại: logs/eval_results.json")

    if map50 >= 0.70:
        log("  ✅ Đạt mục tiêu mAP@0.5 ≥ 0.70 — Model sẵn sàng dùng!")
    else:
        warn(f"  mAP = {map50:.3f} < 0.70. Xem gợi ý cải thiện bên dưới.")
        _print_improvement_tips(map50)


def _print_improvement_tips(current_map):
    tips = [
        "Thu thập thêm ảnh cho class có mAP thấp (≥ 500 instances)",
        "Tăng epochs lên 150–200 và giảm patience xuống 30",
        "Thử backbone lớn hơn: yolov8m.pt thay vì yolov8s.pt",
        "Thêm augmentation: mosaic, mixup",
        "Kiểm tra nhãn sai: dùng prepare_dataset.py --verify để review",
        "Giảm batch size nếu bị OOM, tăng nếu còn dư VRAM",
    ]
    print("\n  Gợi ý cải thiện:")
    for i, tip in enumerate(tips, 1):
        print(f"    {i}. {tip}")


# ══════════════════════════════════════════════════════════
#  BƯỚC 3: XUẤT ONNX
# ══════════════════════════════════════════════════════════
def export_model():
    section("BƯỚC 3 — Xuất model ONNX")

    try:
        from ultralytics import YOLO
    except ImportError:
        return

    best_pt = Path(WEIGHTS_DIR) / "best.pt"
    if not best_pt.exists():
        # Tìm best.pt trong runs/
        candidates = list(Path("runs").rglob("best.pt")) if Path("runs").exists() else []
        if not candidates:
            warn("Chưa có best.pt. Hãy huấn luyện trước.")
            return
        # Sắp xếp theo thời gian sửa đổi mới nhất
        candidates.sort(key=lambda p: p.stat().st_mtime)
        found_pt = candidates[-1]
        info(f"Tìm thấy best.pt tại: {found_pt}. Đang copy vào {best_pt}...")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        shutil.copy2(found_pt, best_pt)

    model = YOLO(str(best_pt))
    model.export(format="onnx", imgsz=640, opset=12, simplify=True)
    onnx_path = best_pt.with_suffix(".onnx")
    if onnx_path.exists():
        dst = Path("models/onnx/best.onnx")
        os.makedirs("models/onnx", exist_ok=True)
        shutil.copy2(onnx_path, dst)
        size_mb = dst.stat().st_size / (1024**2)
        log(f"  ✅ Xuất ONNX thành công: {dst}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv8")
    parser.add_argument("--train",     action="store_true", help="Huấn luyện model")
    parser.add_argument("--evaluate",  action="store_true", help="Đánh giá model")
    parser.add_argument("--export",    action="store_true", help="Xuất ONNX")
    parser.add_argument("--all",       action="store_true", help="Chạy toàn bộ pipeline")
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}")
    print("╔══════════════════════════════════════════════════╗")
    print("║   HUẤN LUYỆN YOLOV8                              ║")
    print("╚══════════════════════════════════════════════════╝")
    print(RESET)

    if args.all or args.train:
        train_model()
    if args.all or args.evaluate:
        evaluate_model()
    if args.all or args.export:
        export_model()

    if not any(vars(args).values()):
        print("  Cách dùng:")
        print("    python scripts/train.py --train       # Huấn luyện")
        print("    python scripts/train.py --evaluate    # Đánh giá")
        print("    python scripts/train.py --export      # Xuất ONNX")
        print("    python scripts/train.py --all         # Chạy tất cả")


if __name__ == "__main__":
    main()

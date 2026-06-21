"""
=============================================================
SCRIPT: Thiết lập cấu trúc dự án
Hệ thống cảnh báo va chạm (Collision Warning System)
=============================================================
"""

import os
import subprocess
import sys
import platform

# ── Màu terminal ───────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def log(msg, color=GREEN):   print(f"{color}{msg}{RESET}")
def warn(msg):               print(f"{YELLOW}⚠  {msg}{RESET}")
def error(msg):              print(f"{RED}✗  {msg}{RESET}")
def section(title):          print(f"\n{BOLD}{CYAN}{'='*55}\n  {title}\n{'='*55}{RESET}")

# ── Cấu trúc thư mục dự án ────────────────────────────────
PROJECT_DIRS = [
    "data/splits/train/images",
    "data/splits/train/labels",
    "data/splits/val/images",
    "data/splits/val/labels",
    "data/splits/test/images",
    "data/splits/test/labels",
    "models/weights",
    "models/onnx",
    "src/distance",
    "src/tracking",
    "src/alert",
    "src/ui",
    "src/utils",
    "demo/videos",
    "demo/outputs",
    "docs",
    "scripts",
    "logs",
    "configs",
]

# ── Thư viện cần cài đặt ──────────────────────────────────
REQUIREMENTS = """# Hệ thống cảnh báo va chạm - Requirements
# Cài đặt: pip install -r requirements.txt

# Deep Learning
torch>=2.0.0
torchvision>=0.15.0

# Object Detection
ultralytics>=8.0.0          # YOLOv8

# Computer Vision
opencv-python>=4.8.0
Pillow>=9.0.0

# Data & Dataset
roboflow>=1.0.0

# Tracking
filterpy>=1.4.5             # Kalman filter cho SORT

# Audio Alert
pygame>=2.0.0
gTTS>=2.3.0                 # Google Text-to-Speech (TTS)
pyttsx3>=2.90               # Offline TTS backup

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
tqdm>=4.65.0
numpy>=1.24.0
pandas>=2.0.0
PyYAML>=6.0
"""

# ── File cấu hình data.yaml ───────────────────────────────
DATA_YAML = """# YOLOv8 Dataset Configuration
# Hệ thống cảnh báo va chạm

path: ./data/splits          # Thư mục gốc dataset
train: train/images
val:   val/images
test:  test/images

# Số lượng class
nc: 4

# Tên các class (thứ tự PHẢI khớp với label trong Roboflow)
names:
  0: car           # Ô tô
  1: motorbike     # Xe máy
  2: person        # Người đi bộ
  3: truck         # Xe tải
"""

# ── File cấu hình hệ thống ────────────────────────────────
SYSTEM_CONFIG = """# ============================================================
# Cấu hình hệ thống cảnh báo va chạm
# ============================================================

detection:
  model_path: models/weights/best.pt
  confidence: 0.50            # Ngưỡng confidence tối thiểu
  iou_threshold: 0.45
  img_size: 640
  device: auto                # auto | cpu | cuda | mps

distance:
  # Thông số camera hành trình (cần hiệu chỉnh)
  focal_length_px: 700        # Tiêu cự tính theo pixel (hiệu chỉnh sau)
  # Chiều rộng thực của đối tượng (mét)
  real_widths:
    car: 1.8
    motorbike: 0.7
    person: 0.5
    truck: 2.5
    obstacle: 0.5

ttc:
  history_frames: 10          # Số frame để tính vận tốc trung bình
  min_approach_speed: 0.1     # m/s - bỏ qua nếu chậm hơn

alert:
  # Ngưỡng TTC (giây)
  safe_threshold: 5.0
  warning_threshold: 2.0
  # Khoảng cách tuyệt đối (mét)
  safe_distance: 10.0
  warning_distance: 5.0
  danger_distance: 2.0
  # Cooldown giữa các cảnh báo âm thanh (giây)
  audio_cooldown: 2.0
  tts_language: vi            # Ngôn ngữ TTS: vi = tiếng Việt

output:
  show_fps: true
  show_distance: true
  show_ttc: true
  save_video: false
  log_csv: true
  log_path: logs/events.csv
"""

# ── README.md ─────────────────────────────────────────────
README = """# 🚨 Hệ thống cảnh báo va chạm (Collision Warning System)

Hệ thống phát hiện vật thể và cảnh báo va chạm theo thời gian thực từ camera hành trình.

## Kiến trúc tổng thể

```
Video/Camera → YOLOv8 Detection → Object Tracking
                                       ↓
                               Distance Estimation (Pinhole Model)
                                       ↓
                               TTC Calculation
                                       ↓
                        ┌──────────────────────────┐
                        │  SAFE | WARNING | DANGER │
                        └──────────────────────────┘
                                       ↓
                          Visual Overlay + Audio TTS
```

## Cấu trúc thư mục

```
collision_warning_system/
├── data/
│   └── splits/         # train / val / test
├── models/
│   ├── weights/        # File .pt đã huấn luyện
│   └── onnx/           # Model xuất ONNX
├── src/
│   ├── distance/       # Ước lượng khoảng cách
│   ├── tracking/       # Object tracker
│   ├── alert/          # Logic cảnh báo + TTS
│   ├── ui/             # Giao diện hiển thị dashboard
│   └── utils/          # Tiện ích chung
├── demo/               # Video demo và kết quả
├── docs/               # Tài liệu kỹ thuật
├── scripts/            # Scripts hỗ trợ (setup, prepare, train...)
├── configs/
│   ├── data.yaml       # Cấu hình dataset YOLOv8
│   └── system.yaml     # Cấu hình hệ thống
├── logs/               # File log CSV
├── requirements.txt
└── main.py             # Entry point chính
```

## Cài đặt

```bash
# Clone repo
git clone <your-repo-url>
cd collision_warning_system

# Cài đặt thư viện
pip install -r requirements.txt

# Chạy demo
python main.py --source demo/videos/test.mp4
```

## Lộ trình phát triển

| Bước | Nhiệm vụ | Trạng thái |
|------|----------|-----------|
| 1    | Thiết lập môi trường & cấu trúc | ✅ |
| 3    | Gán nhãn & chuẩn bị dataset     | ⏳ |
| 4    | Huấn luyện YOLOv8               | ⏳ |
| 5    | Ước lượng khoảng cách           | ⏳ |
| 6    | Tính toán TTC                   | ⏳ |
| 7    | Logic cảnh báo & âm thanh       | ⏳ |
| 8    | Tích hợp pipeline               | ⏳ |
| 9    | Giao diện demo                  | ⏳ |
| 10   | Kiểm thử & hoàn thiện           | ⏳ |

## Công nghệ sử dụng

- **YOLOv8** (Ultralytics) — Object Detection
- **OpenCV** — Video processing, Computer Vision
- **PyTorch** — Deep Learning framework
- **Roboflow** — Gán nhãn & augmentation
- **gTTS / pyttsx3** — Text-to-Speech cảnh báo
- **matplotlib** — Trực quan hoá kết quả
"""

# ── Pipeline diagram (main.py skeleton) ───────────────────
MAIN_PY = """\"\"\"
main.py - Entry point hệ thống cảnh báo va chạm
Chạy: python main.py --source <video_path_or_camera_index>
\"\"\"

import argparse
import sys
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Collision Warning System")
    parser.add_argument("--source",   default="0",
                        help="Nguồn video: path file hoặc index camera (0=webcam)")
    parser.add_argument("--config",   default="configs/system.yaml",
                        help="File cấu hình hệ thống")
    parser.add_argument("--no-audio", action="store_true",
                        help="Tắt cảnh báo âm thanh")
    parser.add_argument("--save",     action="store_true",
                        help="Lưu video kết quả")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 55)
    print("  HỆ THỐNG CẢNH BÁO VA CHẠM")
    print("  Collision Warning System v1.0")
    print("=" * 55)
    print(f"  Nguồn video : {args.source}")
    print(f"  Cấu hình    : {args.config}")
    print(f"  Âm thanh    : {'Tắt' if args.no_audio else 'Bật'}")
    print("=" * 55)

    # TODO: from src.distance.estimator import DistanceEstimator
    # TODO: from src.tracking.ttc import ObjectTracker
    # TODO: from src.alert.alert_system import AlertSystem
    # TODO: from src.ui.dashboard import ... (Giao diện hiển thị)
    # TODO: Pipeline tích hợp đầy đủ

    print("\\n[INFO] Pipeline chưa được tích hợp.")
    print("[INFO] Kiểm tra cấu trúc thư mục và configs/ để bắt đầu.")


if __name__ == "__main__":
    main()
"""

# ── Hàm kiểm tra GPU / CUDA ───────────────────────────────
CHECK_GPU = """\"\"\"
scripts/check_gpu.py - Kiểm tra thiết bị GPU/CUDA
\"\"\"

def check_device():
    print("\\n== KIỂM TRA THIẾT BỊ ==")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        mps_ok  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✅ CUDA GPU   : {name} ({mem:.1f} GB VRAM)")
            print(f"     CUDA ver  : {torch.version.cuda}")
            print(f"  → Dùng device='cuda' khi huấn luyện")
        elif mps_ok:
            print("  ✅ Apple MPS  : Phát hiện GPU Apple Silicon")
            print("  → Dùng device='mps' khi huấn luyện")
        else:
            print("  ⚠  CPU only  : Không có GPU khả dụng")
            print("  → Huấn luyện sẽ rất chậm. Khuyến nghị dùng Google Colab.")

        print(f"  PyTorch       : {torch.__version__}")
    except ImportError:
        print("  ✗  PyTorch chưa được cài đặt!")
        print("     Chạy: pip install torch torchvision")

    try:
        import cv2
        print(f"  OpenCV        : {cv2.__version__} ✅")
    except ImportError:
        print("  ✗  OpenCV chưa được cài đặt!")

    print()

if __name__ == "__main__":
    check_device()
"""

# ══════════════════════════════════════════════════════════
#  HÀM CHÍNH
# ══════════════════════════════════════════════════════════
def create_project_structure():
    section("BƯỚC 1 — Tạo cấu trúc thư mục")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for d in PROJECT_DIRS:
        path = os.path.join(base, d)
        os.makedirs(path, exist_ok=True)
        # Thêm .gitkeep để git theo dõi thư mục rỗng
        gitkeep = os.path.join(path, ".gitkeep")
        if not os.listdir(path):
            open(gitkeep, "w").close()
        log(f"  ✓ {d}/")

    log("\n  Cấu trúc thư mục đã tạo xong!")


def write_config_files():
    section("BƯỚC 2 — Tạo file cấu hình")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    files = {
        "requirements.txt":   REQUIREMENTS,
        "configs/data.yaml":  DATA_YAML,
        "configs/system.yaml": SYSTEM_CONFIG,
        "README.md":          README,
        "main.py":            MAIN_PY,
        "scripts/check_gpu.py": CHECK_GPU,
    }

    for name, content in files.items():
        path = os.path.join(base, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        log(f"  ✓ {name}")

    log("\n  File cấu hình đã tạo xong!")


def verify_setup():
    section("BƯỚC 3 — Kiểm tra thiết bị")
    import importlib

    libs = [
        ("cv2",         "OpenCV"),
        ("numpy",       "NumPy"),
        ("matplotlib",  "Matplotlib"),
        ("yaml",        "PyYAML"),
    ]

    all_ok = True
    for module, name in libs:
        try:
            m = importlib.import_module(module)
            ver = getattr(m, "__version__", "?")
            log(f"  ✅ {name:<15} v{ver}")
        except ImportError:
            warn(f"  ✗  {name:<15} chưa được cài đặt")
            all_ok = False

    # Kiểm tra riêng torch (optional nhưng cần cho training)
    try:
        import torch
        cuda = "CUDA ✅" if torch.cuda.is_available() else "CPU only"
        log(f"  ✅ PyTorch          v{torch.__version__} — {cuda}")
    except ImportError:
        warn("  ⚠  PyTorch chưa cài — cần cho bước huấn luyện (Training)")

    if all_ok:
        log("\n  Môi trường cơ bản sẵn sàng!", CYAN)
    else:
        warn("\n  Một số thư viện còn thiếu. Chạy: pip install -r requirements.txt")


def print_next_steps():
    section("HOÀN THÀNH THIẾT LẬP ✅")
    print(f"""
  Đã tạo:
    ✓ Cấu trúc thư mục 20+ thư mục
    ✓ requirements.txt
    ✓ configs/data.yaml
    ✓ configs/system.yaml
    ✓ README.md
    ✓ main.py (skeleton)
    ✓ scripts/check_gpu.py

  Bước tiếp theo:
    → Chuẩn bị và kiểm tra Dataset với scripts/prepare_dataset.py

  Commit gợi ý:
    git init && git add . && git commit -m "Project structure & environment setup"
{RESET}""")


if __name__ == "__main__":
    print(f"{BOLD}{CYAN}")
    print("╔══════════════════════════════════════════════════╗")
    print("║   HỆ THỐNG CẢNH BÁO VA CHẠM - THIẾT LẬP          ║")
    print("╚══════════════════════════════════════════════════╝")
    print(RESET)

    create_project_structure()
    write_config_files()
    verify_setup()
    print_next_steps()

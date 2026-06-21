# 🚨 Hệ thống cảnh báo va chạm (Collision Warning System)

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
                        │  SAFE | WARNING | DANGER  │
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
| 3    | Gán nhãn & chuẩn bị dataset     | ✅ |
| 4    | Huấn luyện YOLOv8               | ✅ |
| 5    | Ước lượng khoảng cách           | ✅ |
| 6    | Tính toán TTC                   | ✅ |
| 7    | Logic cảnh báo & âm thanh       | ✅ |
| 8    | Tích hợp pipeline               | ✅ |
| 9    | Giao diện demo                  | ✅ |
| 10   | Kiểm thử & hoàn thiện           | ✅ |

## Công nghệ sử dụng

- **YOLOv8** (Ultralytics) — Object Detection
- **OpenCV** — Video processing, Computer Vision
- **PyTorch** — Deep Learning framework
- **Roboflow** — Gán nhãn & augmentation
- **gTTS / pyttsx3** — Text-to-Speech cảnh báo
- **matplotlib** — Trực quan hoá kết quả

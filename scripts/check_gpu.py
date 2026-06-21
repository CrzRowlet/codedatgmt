"""
scripts/check_gpu.py - Kiểm tra thiết bị GPU/CUDA
"""

def check_device():
    print("\n== KIỂM TRA THIẾT BỊ ==")
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

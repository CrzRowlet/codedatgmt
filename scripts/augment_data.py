"""
Script: Tẩy rửa và Augmentation dữ liệu từ D:\\Dataset_labeled sang D:\\codedatgmt\\data\\splits
"""
import os
import sys
import cv2
import shutil
from pathlib import Path
import random

# Đảm bảo in UTF-8 trên Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

try:
    import albumentations as A
except ImportError:
    print("Vui lòng cài đặt albumentations: pip install albumentations")
    exit(1)

# Cấu hình
SRC_DIR = r"D:\Dataset_labeled"
DST_DIR = r"D:\codedatgmt\data\splits"
NUM_AUGMENTS = 2 # Mỗi ảnh tạo thêm 2 bản aug -> tổng 3x

# Mapping class
# Cũ: 0:bicycle, 1:bus, 2:car, 3:motorbike, 4:person, 5:truck
# Mới: 0:car, 1:motorbike, 2:person, 3:truck
CLASS_MAPPING = {
    0: 1, # bicycle -> motorbike
    1: 3, # bus -> truck
    2: 0, # car -> car
    3: 1, # motorbike -> motorbike
    4: 2, # person -> person
    5: 3  # truck -> truck
}

# Augmentation Pipeline cho tập Train
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0, p=0.5),
    A.Blur(blur_limit=(3, 3), p=0.3),
    A.GaussNoise(p=0.3),
    A.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0), p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

def parse_yolo_label(label_path):
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            if cls_id in CLASS_MAPPING:
                new_cls = CLASS_MAPPING[cls_id]
                bboxes.append([cx, cy, w, h])
                class_labels.append(new_cls)
    return bboxes, class_labels

def save_yolo_label(out_path, bboxes, class_labels):
    with open(out_path, 'w') as f:
        for bbox, cls_id in zip(bboxes, class_labels):
            cx, cy, w, h = bbox
            f.write(f"{int(float(cls_id))} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def process_dataset():
    src_images = Path(SRC_DIR) / "images"
    src_labels = Path(SRC_DIR) / "labels"
    
    # Lấy tất cả ảnh gốc .jpg
    image_files = list(src_images.glob("*.jpg"))
    if not image_files:
        print(f"Không tìm thấy ảnh .jpg nào trong: {src_images}")
        return
        
    print(f"Đang quét lọc dữ liệu nhãn hợp lệ từ {len(image_files)} ảnh gốc...")
    
    # Chỉ giữ lại các ảnh có nhãn hợp lệ (sau khi remap class)
    valid_images = []
    for img_path in image_files:
        label_path = src_labels / (img_path.stem + ".txt")
        bboxes, class_labels = parse_yolo_label(label_path)
        if bboxes:
            valid_images.append((img_path, bboxes, class_labels))
            
    total_valid = len(valid_images)
    print(f"Đã lọc được {total_valid} ảnh có nhãn hợp lệ để thực hiện phân chia splits.")
    
    if total_valid == 0:
        print("Không tìm thấy dữ liệu hợp lệ để xử lý!")
        return
        
    # Xáo trộn ngẫu nhiên để chia tập
    random.seed(42)  # Đảm bảo kết quả chia nhất quán
    random.shuffle(valid_images)
    
    # Chia tỷ lệ 80% Train / 10% Val / 10% Test
    train_end = int(total_valid * 0.8)
    val_end = int(total_valid * 0.9)
    
    splits = {
        "train": valid_images[:train_end],
        "val":   valid_images[train_end:val_end],
        "test":  valid_images[val_end:]
    }
    
    # Khởi tạo đếm thống kê kết quả
    stats = {
        "train": {"orig": 0, "aug": 0},
        "val":   {"orig": 0, "aug": 0},
        "test":  {"orig": 0, "aug": 0}
    }
    
    for split_name, dataset in splits.items():
        dst_images = Path(DST_DIR) / split_name / "images"
        dst_labels = Path(DST_DIR) / split_name / "labels"
        
        os.makedirs(dst_images, exist_ok=True)
        os.makedirs(dst_labels, exist_ok=True)
        
        print(f"\n[Splits] Đang xử lý tập {split_name.upper()} ({len(dataset)} ảnh gốc)...")
        
        processed = 0
        for img_path, bboxes, class_labels in dataset:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # 1. Lưu ảnh gốc đã remap class vào split đích
            base_name = img_path.stem
            out_img_path = dst_images / f"{base_name}_orig.jpg"
            out_lbl_path = dst_labels / f"{base_name}_orig.txt"
            cv2.imwrite(str(out_img_path), img)
            save_yolo_label(out_lbl_path, bboxes, class_labels)
            stats[split_name]["orig"] += 1
            
            # 2. Thực hiện Augmentation cho tất cả các tập (train, val, test)
            if NUM_AUGMENTS > 0:
                for i in range(NUM_AUGMENTS):
                    try:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w = img.shape[:2]
                        
                        # Re-init transform động theo kích thước ảnh thực tế để đảm bảo an toàn khi crop
                        custom_transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0, p=0.5),
                            A.Blur(blur_limit=(3, 3), p=0.3),
                            A.GaussNoise(p=0.3),
                            A.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0), p=0.3)
                        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
                        
                        augmented = custom_transform(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
                        aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                        aug_bboxes = augmented['bboxes']
                        aug_labels = augmented['class_labels']
                        
                        if aug_bboxes:
                            out_aug_img = dst_images / f"{base_name}_aug_{i}.jpg"
                            out_aug_lbl = dst_labels / f"{base_name}_aug_{i}.txt"
                            cv2.imwrite(str(out_aug_img), aug_img)
                            save_yolo_label(out_aug_lbl, aug_bboxes, aug_labels)
                            stats[split_name]["aug"] += 1
                    except Exception as e:
                        continue
                        
            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(dataset)}...")
                
    print("\n" + "="*55)
    print("  HOÀN THÀNH XỬ LÝ AUGMENTATION & CHIA DATASET SPLITS! ✅")
    print("="*55)
    for name in ["train", "val", "test"]:
        orig = stats[name]["orig"]
        aug = stats[name]["aug"]
        print(f"  • {name.upper():<6}: {orig:>4} ảnh gốc | {aug:>4} ảnh augmented | Tổng = {orig + aug}")
    print("="*55 + "\n")

if __name__ == "__main__":
    process_dataset()

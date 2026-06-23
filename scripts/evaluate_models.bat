@echo off
echo ==============================================================
echo   TU DONG DANH GIA (VALIDATION) CA 2 MO HINH YOLO
echo ==============================================================
echo.
echo [1/2] Dang chay danh gia mo hinh Fine-Tuning...
echo Ket qua se duoc ghi de vao: runs\detect\val_finetune
call .venv\Scripts\yolo val model=models\weights\best.pt data=configs\data.yaml name=val_finetune project=runs\detect exist_ok=True

echo.
echo [2/2] Dang chay danh gia mo hinh Train From Scratch...
echo Ket qua se duoc ghi de vao: runs\detect\val_scratch
call .venv\Scripts\yolo val model=scratch\Yolo_from_scratch\best.pt data=configs\data.yaml name=val_scratch project=runs\detect exist_ok=True

echo.
echo HOAN TAT! Moi ban vao thu muc runs\detect\ de xem bieu do.
pause

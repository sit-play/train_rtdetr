"""
YOLO Segmentation Training Script
อ่านการตั้งค่าจาก yolo_train_config.yaml
"""

from ultralytics import RTDETR
import yaml
from pathlib import Path
import sys


def train_yolo_from_config(config_path="yolo_train_config.yaml"):
    """
    เทรน YOLO โดยอ่านค่าจาก config file
    
    Parameters:
    -----------
    config_path : str
        path ของ config file (default: yolo_train_config.yaml)
    """
    
    # โหลด config
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"❌ ไม่พบไฟล์: {config_path}")
        print(f"💡 สร้างไฟล์ yolo_train_config.yaml ก่อนแล้วรันใหม่")
        sys.exit(1)
    
    print(f"✅ โหลด config จาก: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # ========== อ่านค่าจาก config ==========
    
    # Dataset
    DATA_PATH = config['dataset']['data_path']
    
    # Model
    MODEL = config['model']['name']
    
    # Training
    EPOCHS = config['training']['epochs']
    BATCH_SIZE = config['training']['batch_size']
    IMAGE_SIZE = config['training']['image_size']
    DEVICE = config['training']['device']
    PATIENCE = config['training']['patience']
    SEED = config['training']['seed']
    
    # Project
    PROJECT = config['project']['save_dir']
    NAME = config['project']['name']
    
    # Augmentation - Geometric
    geo = config['augmentation']['geometric']
    DEGREES = geo['degrees'] if geo['enabled'] else 0.0
    TRANSLATE = geo['translate'] if geo['enabled'] else 0.0
    SCALE = geo['scale'] if geo['enabled'] else 0.0
    SHEAR = geo['shear'] if geo['enabled'] else 0.0
    PERSPECTIVE = geo['perspective'] if geo['enabled'] else 0.0
    FLIPUD = geo['flipud'] if geo['enabled'] else 0.0
    FLIPLR = geo['fliplr'] if geo['enabled'] else 0.0
    MOSAIC = geo['mosaic'] if geo['enabled'] else 0.0
    MIXUP = geo['mixup'] if geo['enabled'] else 0.0
    
    # Augmentation - Color
    color = config['augmentation']['color']
    HSV_H = color['hsv_h'] if color['enabled'] else 0.0
    HSV_S = color['hsv_s'] if color['enabled'] else 0.0
    HSV_V = color['hsv_v'] if color['enabled'] else 0.0
    
    # Hyperparameters
    LEARNING_RATE = config['hyperparameters']['learning_rate']
    MOMENTUM = config['hyperparameters']['momentum']
    WEIGHT_DECAY = config['hyperparameters']['weight_decay']
    WARMUP_EPOCHS = config['hyperparameters']['warmup_epochs']
    BOX = config['hyperparameters']['box']
    CLS = config['hyperparameters']['cls']
    DFL = config['hyperparameters']['dfl']
    
    # Advanced
    WORKERS = config['advanced']['workers']
    AMP = config['advanced']['amp']
    DETERMINISTIC = config['advanced']['deterministic']
    OPTIMIZER = config['advanced']['optimizer']
    PLOTS = config['advanced']['plots']
    VERBOSE = config['advanced']['verbose']
    
    # ========== แสดงข้อมูล ==========
    
    print("\n" + "="*70)
    print("🚀 YOLO Segmentation Training")
    print("="*70)
    print(f"Dataset: {DATA_PATH}")
    print(f"Model: {MODEL}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print("="*70)
    
    print("\n📊 Augmentation Settings:")
    if geo['enabled']:
        print("Geometric Augmentation: ON ✅")
        print(f"  - Degrees: {DEGREES}")
        print(f"  - Translate: {TRANSLATE}")
        print(f"  - Scale: {SCALE}")
        print(f"  - FlipLR: {FLIPLR}")
        print(f"  - FlipUD: {FLIPUD}")
        print(f"  - Mosaic: {MOSAIC}")
    else:
        print("Geometric Augmentation: OFF ❌")
    
    if color['enabled']:
        print("\nColor Augmentation: ON ✅")
        print(f"  - HSV-H: {HSV_H}")
        print(f"  - HSV-S: {HSV_S}")
        print(f"  - HSV-V: {HSV_V}")
    else:
        print("\nColor Augmentation: OFF ❌")
    
    print("="*70)
    
    # ========== เทรน ==========
    
    print("\n🔄 กำลังโหลด model...")
    model = RTDETR(MODEL)
    
    print("🚀 เริ่มการเทรน...\n")
    results = model.train(
        # Dataset
        data=DATA_PATH,
        
        # Training config
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        
        # Project config
        project=PROJECT,
        name=NAME,
        exist_ok=False,
        
        # Augmentation - Geometric
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=0.0,
        auto_augment=None,
        erasing=0.0,
        
        # Augmentation - Color
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        
        # Hyperparameters
        lr0=LEARNING_RATE,
        lrf=0.01,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=BOX,
        cls=CLS,
        dfl=DFL,
        
        # Other settings
        patience=PATIENCE,
        save_period=-1,
        workers=WORKERS,
        seed=SEED,
        deterministic=DETERMINISTIC,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=0,
        amp=AMP,
        fraction=1.0,
        profile=False,
        freeze=None,
        optimizer=OPTIMIZER,
        verbose=VERBOSE,
        plots=PLOTS,
        save=True,
        save_txt=False,
        save_hybrid=False,
        cache=False,
        val=True,
        split='val',
        resume=False,
        pretrained=False,
    )
    
    # ========== แสดงผลลัพธ์ ==========
    
    print("\n" + "="*70)
    print("✅ Training เสร็จสิ้น!")
    print("="*70)
    print(f"\n📁 Results saved to: {PROJECT}/{NAME}")
    print(f"🏆 Best model: {PROJECT}/{NAME}/weights/best.pt")
    print(f"📊 Metrics: {PROJECT}/{NAME}/results.csv")
    
    # Metrics
    print("\n" + "="*70)
    print("📈 Training Results:")
    print("="*70)
    
    metrics = results.results_dict
    if metrics:
        print(f"Box mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"Box mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"Mask mAP50: {metrics.get('metrics/mAP50(M)', 'N/A'):.4f}")
        print(f"Mask mAP50-95: {metrics.get('metrics/mAP50-95(M)', 'N/A'):.4f}")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    
    # ========== วิธีใช้งาน ==========
    # 1. แก้ไฟล์ yolo_train_config.yaml
    # 2. รันโปรแกรม: python train_yolo_seg.py
    
    # ระบุ path ของ config file (optional)
    config_file = "/home/play/Desktop/train_rtdetr/config/train_yolo_config.yaml"
    
    # ถ้าต้องการใช้ config file อื่น
    # config_file = "my_custom_config.yaml"
    
    # เทรน
    results = train_yolo_from_config(config_file)
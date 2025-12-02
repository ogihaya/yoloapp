import os
import sys
from pathlib import Path


import torch
from PIL import Image
from loguru import logger
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
print (project_root)

from yolo import create_model
from torch2trt import torch2trt
import tensorrt as trt

if __name__ == '__main__':

    LOG_LEVEL = trt.Logger.WARNING


    MODEL = "v9-s"
    DEVICE = "cuda:0"
    WEIGHT_PATH = f"weights/test-model.pt"
    TRT_WEIGHT_PATH = f"weights/test-model.trt"
    MODEL_CONFIG = f"yolo/config/model/{MODEL}.yaml"


    IMAGE_SIZE = (960, 960)
    NUM_CLASSES = 2  # カスタムデータセットのクラス数


    device = torch.device(DEVICE)


    # モデル設定を読み込み
    with open(MODEL_CONFIG) as stream:
        cfg_model = OmegaConf.load(stream)



    # モデル作成と重み読み込み
    with torch.no_grad():
        try:
            model = create_model(cfg_model, weight_path=WEIGHT_PATH, class_num=NUM_CLASSES)
            model = model.to(device).eval()
            logger.success(f"✅ Model loaded with {NUM_CLASSES} classes")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            sys.exit(1)


    # ダミー入力でテスト
    dummy_input = torch.ones((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])).to(device)


    # TensorRT変換
    logger.info(f"♻️ Creating TensorRT model...")
    try:
        model_trt = torch2trt(
            model,
            [dummy_input],
            fp16_mode=True
        )
        
        
    except Exception as e:
        logger.error(f"❌ TensorRT conversion failed: {e}")
        sys.exit(1)


    # TensorRTモデルを保存
    torch.save(model_trt.state_dict(), TRT_WEIGHT_PATH)
    logger.success(f"📥 TensorRT model saved to {TRT_WEIGHT_PATH}")

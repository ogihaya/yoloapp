import base64
import hashlib
import importlib
import io
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # Pillow is not installed yet
    Image = None  # type: ignore

    class UnidentifiedImageError(Exception):
        """Fallback so callers can catch a meaningful exception."""


logger = logging.getLogger(__name__)


class YoloInferenceError(RuntimeError):
    """Base exception for inference failures."""


class MissingDependencyError(YoloInferenceError):
    """Raised when YOLO's runtime dependencies are missing."""


class ModelLoadError(YoloInferenceError):
    """Raised when a model checkpoint cannot be hydrated."""


class ImageDecodeError(YoloInferenceError):
    """Raised when an uploaded file is not a valid image."""


@dataclass
class InferenceSettings:
    img_size: int = 640
    min_confidence: float = 0.25
    min_iou: float = 0.45
    max_bbox: int = 300

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ImagePayload:
    id: str
    name: str
    data: bytes


class YoloV9InferenceEngine:
    """Thin wrapper that keeps a single YOLOv9 model instance in memory."""

    def __init__(self) -> None:
        self._project_root = Path(__file__).resolve().parents[3]
        self._yolo_root = self._project_root / "yolov9"
        self._torch = None
        self._compose = None
        self._initialize = None
        self._AugmentationComposer = None
        self._PostProcess = None
        self._create_converter = None
        self._create_model = None
        self._draw_bboxes = None
        self._NMSConfig = None
        self._device = None
        self._cfg = None
        self._model = None
        self._class_names: List[str] = []
        self._current_weight_hash: Optional[str] = None
        self._lock = threading.RLock()
        self._bootstrapped = False

    @property
    def class_names(self) -> List[str]:
        return list(self._class_names)

    @property
    def device_label(self) -> str:
        if self._device is None:
            return "uninitialized"
        label = str(self._device)
        if self._torch is None:
            return label
        if label.startswith("cuda"):
            name = self._torch.cuda.get_device_name(0)
            return f"CUDA ({name})"
        if label == "mps":
            return "Apple MPS"
        return "CPU"

    def run(
        self,
        *,
        model_bytes: bytes,
        settings: InferenceSettings,
        images: List[ImagePayload],
        model_label: str = "uploaded.pt",
    ) -> Dict[str, Any]:
        with self._lock:
            self._ensure_bootstrapped()
            self._load_weights(model_bytes, model_label)
            return self._predict(images, settings)

    # ------------------------------------------------------------------ #
    # Bootstrap helpers
    # ------------------------------------------------------------------ #
    def _ensure_bootstrapped(self) -> None:
        if self._bootstrapped:
            return
        if Image is None:
            raise MissingDependencyError("Pillow がインストールされていません。`pip install Pillow` を実行してください。")

        torch = self._safe_import(
            "torch",
            "PyTorch (torch) が見つかりません。`pip install torch torchvision` などで導入してください。",
        )
        _ = self._safe_import("torchvision", "torchvision が不足しています。PyTorchと一緒にインストールしてください。")

        hydra_mod = self._safe_import("hydra", "hydra-core が見つかりません。`pip install hydra-core omegaconf` を実行してください。")
        compose = getattr(hydra_mod, "compose", None)
        initialize = getattr(hydra_mod, "initialize", None)
        if compose is None or initialize is None:
            raise MissingDependencyError("hydra-core のバージョンが古いため compose/initialize を利用できません。")

        # Ensure yolov9 modules are importable
        if str(self._yolo_root) not in sys.path:
            sys.path.append(str(self._yolo_root))

        try:
            yolo_pkg = importlib.import_module("yolo")
        except ImportError as exc:
            raise MissingDependencyError(
                "yolov9 モジュールを読み込めません。`pip install -r yolov9/requirements.txt` の実行を検討してください。"
                f" 原因: {exc}"
            ) from exc

        from yolo.config.config import NMSConfig  # type: ignore

        self._torch = torch
        self._compose = compose
        self._initialize = initialize
        self._NMSConfig = NMSConfig
        self._AugmentationComposer = getattr(yolo_pkg, "AugmentationComposer")
        self._PostProcess = getattr(yolo_pkg, "PostProcess")
        self._create_converter = getattr(yolo_pkg, "create_converter")
        self._create_model = getattr(yolo_pkg, "create_model")
        self._draw_bboxes = getattr(yolo_pkg, "draw_bboxes")

        config_dir = self._yolo_root / "yolo" / "config"
        if not config_dir.exists():
            raise MissingDependencyError(f"YOLOの設定ディレクトリが見つかりません: {config_dir}")

        initialize_kwargs: Dict[str, Any] = {"config_path": str(config_dir), "version_base": None, "job_name": "django_inference"}
        with self._initialize(**initialize_kwargs):
            cfg = self._compose(config_name="config", overrides=["task=inference", "model=v9-s"])

        self._cfg = cfg
        self._device = self._select_device(torch)
        class_num = getattr(cfg.dataset, "class_num", len(getattr(cfg.dataset, "class_list", [])))
        self._model = self._create_model(cfg.model, class_num=class_num).to(self._device)
        self._model.eval()
        self._class_names = list(getattr(cfg.dataset, "class_list", []))
        self._bootstrapped = True
        logger.info("YOLOv9 推論エンジンを初期化しました（device=%s）", self.device_label)

    def _select_device(self, torch_mod):
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        if getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
            return torch_mod.device("mps")
        return torch_mod.device("cpu")

    def _safe_import(self, module_name: str, error_message: str):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise MissingDependencyError(error_message) from exc

    # ------------------------------------------------------------------ #
    # Prediction internals
    # ------------------------------------------------------------------ #
    def _predict(self, images: List[ImagePayload], settings: InferenceSettings) -> Dict[str, Any]:
        if not images:
            raise YoloInferenceError("推論対象の画像がありません。")
        torch = self._torch
        converter = self._create_converter(
            getattr(self._cfg.model, "name", "v9-s"),
            self._model,
            self._cfg.model.anchor,
            [settings.img_size, settings.img_size],
            self._device,
            True,
        )
        nms_cfg = self._NMSConfig(
            min_confidence=settings.min_confidence,
            min_iou=settings.min_iou,
            max_bbox=settings.max_bbox,
        )
        post_process = self._PostProcess(converter, nms_cfg)
        transform = self._AugmentationComposer([], [settings.img_size, settings.img_size])

        total_time_ms = 0.0
        results: List[Dict[str, Any]] = []
        for payload in images:
            pil_image = self._load_image(payload)
            tensor, _, rev_tensor = transform(pil_image)
            tensor = tensor.to(self._device)[None]
            rev_tensor = rev_tensor.to(self._device)[None]

            start = time.perf_counter()
            with torch.no_grad():
                predictions = self._model(tensor)
                detections = post_process(predictions, rev_tensor)
            elapsed_ms = (time.perf_counter() - start) * 1000
            total_time_ms += elapsed_ms

            parsed = self._parse_detections(detections, pil_image.width, pil_image.height)
            annotated = self._draw_bboxes(pil_image, detections, idx2label=self._class_names)
            encoded_image = self._encode_image(annotated)
            results.append(
                {
                    "image_id": payload.id,
                    "filename": payload.name,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "num_detections": len(parsed),
                    "detections": parsed,
                    "result_image": encoded_image,
                    "duration_ms": round(elapsed_ms, 2),
                }
            )

        return {"results": results, "total_time_ms": round(total_time_ms, 2)}

    def _parse_detections(self, detections, width: int, height: int) -> List[Dict[str, Any]]:
        torch = self._torch
        if isinstance(detections, list):
            tensor = detections[0] if detections else torch.zeros((0, 6), device=self._device)
        else:
            tensor = detections
        if tensor is None:
            return []

        tensor = tensor.detach().cpu()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        parsed: List[Dict[str, Any]] = []
        for det in tensor.tolist():
            if len(det) < 6:
                continue
            cls_id = int(det[0])
            bbox = {
                "x1": float(max(0.0, det[1])),
                "y1": float(max(0.0, det[2])),
                "x2": float(min(width, det[3])),
                "y2": float(min(height, det[4])),
            }
            bbox["width"] = max(0.0, bbox["x2"] - bbox["x1"])
            bbox["height"] = max(0.0, bbox["y2"] - bbox["y1"])
            parsed.append(
                {
                    "class_id": cls_id,
                    "class_name": self._class_names[cls_id] if cls_id < len(self._class_names) else f"class_{cls_id}",
                    "confidence": float(det[5]),
                    "bbox": bbox,
                }
            )
        return parsed

    def _load_image(self, payload: ImagePayload):
        try:
            image = Image.open(io.BytesIO(payload.data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except UnidentifiedImageError as exc:
            raise ImageDecodeError(f"画像 {payload.name} を読み込めませんでした。対応形式か確認してください。") from exc

    def _encode_image(self, image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    # ------------------------------------------------------------------ #
    # Model weights
    # ------------------------------------------------------------------ #
    def _load_weights(self, weight_bytes: bytes, label: str) -> None:
        if not weight_bytes:
            raise ModelLoadError("モデルファイルが空です。正しい .pt ファイルを指定してください。")

        weight_hash = hashlib.sha1(weight_bytes).hexdigest()
        if self._current_weight_hash == weight_hash:
            return

        buffer = io.BytesIO(weight_bytes)
        try:
            checkpoint = self._torch.load(buffer, map_location=self._device)
        except Exception as exc:  # pragma: no cover - torch specific
            raise ModelLoadError(f"{label} の読み込みに失敗しました: {exc}") from exc

        state_dict = self._normalize_state_dict(checkpoint)
        try:
            self._model.load_state_dict(state_dict, strict=False)
        except Exception as exc:  # pragma: no cover - torch specific
            raise ModelLoadError(f"モデルパラメータを適用できませんでした: {exc}") from exc

        self._current_weight_hash = weight_hash
        logger.info("モデルウェイトを更新しました: %s", label)

    def _normalize_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                checkpoint = {key.replace("model.model.", ""): value for key, value in checkpoint["state_dict"].items()}
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                checkpoint = checkpoint["model"]
        return checkpoint


_ENGINE: Optional[YoloV9InferenceEngine] = None


def get_engine() -> YoloV9InferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = YoloV9InferenceEngine()
    return _ENGINE

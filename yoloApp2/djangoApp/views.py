import base64
import binascii
import io
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .services.yolo_inference import (
    ImageDecodeError,
    ImagePayload,
    InferenceSettings,
    MissingDependencyError,
    ModelLoadError,
    YoloInferenceError,
    get_engine,
)

logger = logging.getLogger(__name__)


def _normalize_class_names(classes: List[Dict[str, str]]) -> List[str]:
    normalized: List[str] = []
    for index, cls in enumerate(classes):
        label = str(cls.get("label") or "").strip()
        normalized.append(label or f"class_{index}")
    return normalized


def _build_dataset_yaml(class_names: List[str]) -> str:
    class_list_literal = json.dumps(class_names, ensure_ascii=False)
    img_size_literal = json.dumps([1280, 1280])
    lines = [
        "path: dataset",
        "train: train",
        "validation: val",
        "test: test",
        "",
        f"class_num: {len(class_names)}",
        f"class_list: {class_list_literal}",
        "",
        f"img_size: {img_size_literal}",
        "",
        "num_workers: 2",
        "",
    ]
    return "\n".join(lines)

# ホーム画面のビュー関数
def home(request):
    """
    ホーム画面を表示する関数
    request: ブラウザからのリクエスト情報
    """
    return render(request, 'djangoApp/home.html')

# train画面のビュー関数
def train(request):
    """
    train用のアノテーション画面を表示する関数
    """
    return render(request, 'djangoApp/train.html')

# val画面のビュー関数
def val(request):
    """
    val用のアノテーション画面を表示する関数
    """
    return render(request, 'djangoApp/val.html')

# inference画面のビュー関数
def inference(request):
    """
    推論実行画面を表示する関数
    """
    return render(request, 'djangoApp/inference.html')


def _export_dataset_zip(request, dataset_slug: str):
    """
    フロントエンドから送信された画像とアノテーション情報を受け取り、
    指定されたデータセット名でYOLO形式のzipを生成して返す。
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "JSONの形式が正しくありません"}, status=400)

    classes = payload.get("classes") or []
    images = payload.get("images") or []

    if not classes:
        return JsonResponse({"error": "エクスポートする前にクラスを追加してください"}, status=400)
    if not images:
        return JsonResponse({"error": "エクスポートする前に画像を追加してください"}, status=400)

    class_index = {cls["id"]: idx for idx, cls in enumerate(classes)}
    dataset_yaml_content = None
    if dataset_slug.lower() == "train":
        class_names_for_yaml = _normalize_class_names(classes)
        dataset_yaml_content = _build_dataset_yaml(class_names_for_yaml)
    used_names = set()

    def unique_filename(original_name: str, fallback_ext: str = ".png") -> str:
        path = Path(original_name or "image")
        stem = path.stem or "image"
        suffix = path.suffix or fallback_ext
        candidate = f"{stem}{suffix}"
        counter = 1
        while candidate in used_names:
            candidate = f"{stem}_{counter}{suffix}"
            counter += 1
        used_names.add(candidate)
        return candidate

    buffer = io.BytesIO()
    dataset_folder = dataset_slug.strip() or "dataset"

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for image in images:
            data_url = image.get("src")
            if not data_url or "," not in data_url:
                return JsonResponse({"error": f"画像データが無効です: {image.get('name')}"}, status=400)

            header, encoded = data_url.split(",", 1)
            try:
                binary_image = base64.b64decode(encoded)
            except (binascii.Error, ValueError):
                return JsonResponse({"error": f"画像データのデコードに失敗しました: {image.get('name')}"}, status=400)

            fallback_ext = ".jpg" if "jpeg" in header or "jpg" in header else ".png"
            filename = unique_filename(image.get("name", "image"), fallback_ext)
            image_path = f"{dataset_folder}/images/{filename}"
            archive.writestr(image_path, binary_image)

            label_lines = []
            for box in image.get("boxes", []):
                class_id = class_index.get(box.get("classId"))
                if class_id is None:
                    continue

                try:
                    width_pct = float(box.get("w", 0))
                    height_pct = float(box.get("h", 0))
                    left_pct = float(box.get("x", 0))
                    top_pct = float(box.get("y", 0))
                except (TypeError, ValueError):
                    continue

                if width_pct <= 0 or height_pct <= 0:
                    continue

                width = max(0.0, min(1.0, width_pct / 100))
                height = max(0.0, min(1.0, height_pct / 100))
                center_x = max(0.0, min(1.0, (left_pct + width_pct / 2) / 100))
                center_y = max(0.0, min(1.0, (top_pct + height_pct / 2) / 100))

                label_lines.append(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                )

            label_name = f"{dataset_folder}/labels/{Path(filename).stem}.txt"
            label_content = "\n".join(label_lines)
            archive.writestr(label_name, label_content)

        if dataset_yaml_content:
            archive.writestr(f"{dataset_folder}/dataset.yaml", dataset_yaml_content)

    buffer.seek(0)
    response = HttpResponse(buffer.getvalue(), content_type="application/zip")
    response["Content-Disposition"] = f'attachment; filename="{dataset_folder}.zip"'
    return response


@csrf_exempt
@require_POST
def export_train_zip(request):
    """Trainデータセットをzipとして出力する。"""
    return _export_dataset_zip(request, "train")


@csrf_exempt
@require_POST
def export_val_zip(request):
    """Valデータセットをzipとして出力する。"""
    return _export_dataset_zip(request, "val")


def _parse_float(value: Optional[str], default: float, *, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _parse_int(value: Optional[str], default: int, *, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _parse_class_names(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    parsed_names: List[str] = []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            parsed_names = [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        parsed_names = [line.strip() for line in raw.splitlines() if line.strip()]
    return parsed_names or None


def _load_image_metadata(raw: Optional[str]) -> List[Dict[str, str]]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _build_image_payloads(files, metadata: List[Dict[str, str]]) -> List[ImagePayload]:
    payloads: List[ImagePayload] = []
    for index, uploaded in enumerate(files):
        meta = metadata[index] if index < len(metadata) else {}
        image_id = str(meta.get("id") or meta.get("clientId") or index)
        display_name = meta.get("name") or uploaded.name or f"inference_{index + 1}"
        try:
            uploaded.seek(0)
            data = uploaded.read()
        except OSError as exc:
            raise ImageDecodeError(f"画像 {display_name} の読み込みに失敗しました: {exc}") from exc
        payloads.append(ImagePayload(id=image_id, name=display_name, data=data))
    return payloads


def _build_inference_settings(post_data) -> InferenceSettings:
    img_size = _parse_int(post_data.get("img_size"), 640, minimum=32, maximum=2048)
    # YOLOは32の倍数が扱いやすいため丸める
    img_size -= img_size % 32
    if img_size < 32:
        img_size = 32
    min_confidence = _parse_float(post_data.get("min_confidence"), 0.25, minimum=0.0, maximum=1.0)
    min_iou = _parse_float(post_data.get("min_iou"), 0.45, minimum=0.0, maximum=1.0)
    max_bbox = _parse_int(post_data.get("max_bbox"), 300, minimum=1, maximum=2000)
    num_workers = _parse_int(post_data.get("num_workers"), 4, minimum=0, maximum=128)
    class_names = _parse_class_names(post_data.get("class_names"))
    return InferenceSettings(
        img_size=img_size,
        min_confidence=min_confidence,
        min_iou=min_iou,
        max_bbox=max_bbox,
        num_workers=num_workers,
        class_names=class_names,
    )


@csrf_exempt
@require_POST
def run_inference(request):
    """YOLOv9の推論を同期実行しJSONを返却する。"""
    model_file = request.FILES.get("model")
    if not model_file:
        return JsonResponse({"error": "モデルファイル(.pt)をアップロードしてください。"}, status=400)

    image_files = request.FILES.getlist("images")
    if not image_files:
        return JsonResponse({"error": "推論対象の画像を1枚以上追加してください。"}, status=400)

    metadata = _load_image_metadata(request.POST.get("image_metadata", ""))
    try:
        image_payloads = _build_image_payloads(image_files, metadata)
    except ImageDecodeError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    try:
        model_file.seek(0)
    except OSError:
        pass
    model_bytes = model_file.read()
    settings = _build_inference_settings(request.POST)

    engine = get_engine()
    try:
        inference_payload = engine.run(
            model_bytes=model_bytes,
            settings=settings,
            images=image_payloads,
            model_label=model_file.name or "uploaded.pt",
        )
    except MissingDependencyError as exc:
        return JsonResponse({"error": str(exc)}, status=503)
    except ImageDecodeError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except ModelLoadError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except YoloInferenceError as exc:
        logger.exception("YOLO推論でエラーが発生しました")
        return JsonResponse({"error": str(exc)}, status=500)
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("予期せぬ推論エラー")
        return JsonResponse({"error": "推論処理で予期せぬエラーが発生しました。"}, status=500)

    results = inference_payload.get("results", [])
    total_detections = sum(item.get("num_detections", 0) for item in results)
    response_payload = {
        "results": results,
        "device": engine.device_label,
        "class_names": settings.class_names or engine.class_names,
        "settings": settings.as_dict(),
        "stats": {
            "total_images": len(results),
            "total_detections": total_detections,
            "total_time_ms": inference_payload.get("total_time_ms", 0.0),
        },
    }
    return JsonResponse(response_payload)

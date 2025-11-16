import base64
import binascii
import io
import json
import zipfile
from pathlib import Path

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

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

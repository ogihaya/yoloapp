import sys
from pathlib import Path
import os
import glob

import torch
from hydra import compose, initialize
from PIL import Image 

project_root = Path().resolve()
sys.path.append(str(project_root))
print(project_root)

from yolo import (
    AugmentationComposer,
    Config,
    PostProcess,
    create_converter,
    create_model,
    draw_bboxes,
    bbox_nms,
    NMSConfig,
)


def slide_image(image, slide=4, device='cpu'):
    """
    スライディングウィンドウで画像を分割して小物体検出を強化
    
    Args:
        image: 入力画像テンソル [B, C, H, W]
        slide: スライド分割数 (2 or 4推奨)
        device: デバイス
    
    Returns:
        total_image: 分割された画像のテンソル
        total_shift: 各タイルのオフセット座標
    """
    # 画像をスライド倍に拡大
    up_image = torch.nn.functional.interpolate(image, scale_factor=slide, mode='bilinear', align_corners=False)
    
    image_list = [image]  # 元画像も含める
    shift_list = []
    
    *_, w, h = up_image.shape
    
    # スライディングウィンドウで分割
    for x_slide in range(slide):
        for y_slide in range(slide):
            left_w = w // slide * x_slide
            right_w = w // slide * (x_slide + 1)
            left_h = h // slide * y_slide
            right_h = h // slide * (y_slide + 1)
            
            slide_image_crop = up_image[:, :, left_w:right_w, left_h:right_h]
            image_list.append(slide_image_crop)
            shift_list.append(torch.Tensor([left_h, left_w, left_h, left_w]))
    
    total_image = torch.concat(image_list)
    total_shift = torch.stack(shift_list).to(device)
    
    return total_image, total_shift


def cross_class_nms(detections, iou_threshold=0.5):
    """
    異なるクラス間で重複している場合、信頼度が高い方を残す
    
    Args:
        detections: 検出結果テンソル [N, 6] (class_id, x1, y1, x2, y2, confidence)
        iou_threshold: IoU閾値（この値以上で重複とみなす）
    
    Returns:
        filtered_detections: フィルタリング後の検出結果
    """
    if len(detections) == 0:
        return detections
    
    # 信頼度の降順でソート
    confidences = detections[:, 5]
    sorted_indices = torch.argsort(confidences, descending=True)
    sorted_detections = detections[sorted_indices]
    
    keep = []
    
    for i in range(len(sorted_detections)):
        current_det = sorted_detections[i]
        current_class = int(current_det[0].item())
        current_bbox = current_det[1:5]
        
        should_keep = True
        
        # すでに保持されている検出と比較
        for kept_idx in keep:
            kept_det = sorted_detections[kept_idx]
            kept_class = int(kept_det[0].item())
            kept_bbox = kept_det[1:5]
            
            # 異なるクラスの場合のみチェック
            if current_class != kept_class:
                # IoUを計算
                iou = calculate_iou(current_bbox, kept_bbox)
                
                # IoUが閾値以上の場合、信頼度が低い方を削除
                # （既にソート済みなので、現在の検出が信頼度が低い）
                if iou >= iou_threshold:
                    should_keep = False
                    break
        
        if should_keep:
            keep.append(i)
    
    return sorted_detections[keep]


def calculate_iou(box1, box2):
    """
    2つのバウンディングボックスのIoU（Intersection over Union）を計算
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        iou: IoU値
    """
    # 交差領域の座標を計算
    x1_inter = torch.max(box1[0], box2[0])
    y1_inter = torch.max(box1[1], box2[1])
    x2_inter = torch.min(box1[2], box2[2])
    y2_inter = torch.min(box1[3], box2[3])
    
    # 交差領域の面積
    inter_width = torch.clamp(x2_inter - x1_inter, min=0)
    inter_height = torch.clamp(y2_inter - y1_inter, min=0)
    inter_area = inter_width * inter_height
    
    # 各ボックスの面積
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 和集合の面積
    union_area = box1_area + box2_area - inter_area
    
    # IoUを計算
    iou = inter_area / (union_area + 1e-6)  # ゼロ除算を防ぐ
    
    return iou.item()


if __name__ == "__main__":

    CONFIG_PATH = "config"
    CONFIG_NAME = "config"
    MODEL = "v9-s"

    # デバイス設定（MPSが利用可能かチェック）
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        print("MPS is available, using GPU acceleration")
    else:
        DEVICE = "cpu"
        print("MPS not available, using CPU")

    CLASS_NUM = 3  # 11種類のシンボル
    INFERENCE_FOLDER = 'dataset/images/inference/'
    OUTPUT_FOLDER = 'runs/inference/results/'
    CUSTOM_MODEL_PATH = "weights/best.pt"  # 学習済みモデルのパス
    
    # 検出したいクラスを指定（Noneの場合は全クラス、リストで指定する場合はクラスID or クラス名）
    # 例: TARGET_CLASSES = [0, 1, 2]  # クラスIDで指定
    # 例: TARGET_CLASSES = ['class1', 'class2']  # クラス名で指定
    # 例: TARGET_CLASSES = None  # 全クラスを検出
    TARGET_CLASSES = None
    
    # スライディングウィンドウ設定（小物体検出用）
    # SLIDE = 1  # 通常モード（スライディングなし）
    # SLIDE = 2  # 2x2=4タイル（軽量）
    # SLIDE = 4  # 4x4=16タイル（高精度、重い）
    USE_SLIDING_WINDOW = False  # スライディングウィンドウを使用するか（まず通常モードで確認）
    SLIDE = 4  # スライド分割数（2推奨: バランス良好）
    
    # クラス間NMS設定（異なるシンボルが重なった場合の処理）
    USE_CROSS_CLASS_NMS = True  # 異なるクラス間でも重複を削除（信頼度が高い方を残す）
    CROSS_CLASS_IOU_THRESHOLD = 0.5  # 異なるクラス間で重複とみなすIoU閾値
    
    # デバッグモード：すべての検出を詳細に表示
    DEBUG_MODE = True

    # 出力フォルダを作成
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    device = torch.device(DEVICE)

    with initialize(config_path=CONFIG_PATH, version_base=None, job_name="notebook_job"):
        cfg: Config = compose(config_name=CONFIG_NAME, overrides=["task=inference", f"model={MODEL}"])
        print("cfg is")
        print(cfg)
        model = create_model(cfg.model, class_num=CLASS_NUM).to(device)
        cfg.task.nms.min_confidence = 0.02
        cfg.task.nms.min_iou = 0.5
        state_dict = torch.load(CUSTOM_MODEL_PATH, map_location=device)
        new_state_dict = state_dict
        # Load the modified state dictionary into the model
        model.load_state_dict(new_state_dict)

        model.eval()

        transform = AugmentationComposer([], cfg.image_size)

        is_inference = True
        converter = create_converter(cfg.model.name, model, cfg.model.anchor, cfg.image_size, device, is_inference)
        post_proccess = PostProcess(converter, cfg.task.nms)
        
        # TARGET_CLASSESをクラスIDのセットに変換
        target_class_ids = None
        if TARGET_CLASSES is not None:
            target_class_ids = set()
            for cls in TARGET_CLASSES:
                if isinstance(cls, str):
                    # クラス名で指定された場合、クラスIDに変換
                    if cls in cfg.dataset.class_list:
                        target_class_ids.add(cfg.dataset.class_list.index(cls))
                    else:
                        print(f"警告: クラス名 '{cls}' が見つかりませんでした。")
                elif isinstance(cls, int):
                    # クラスIDで指定された場合
                    if 0 <= cls < CLASS_NUM:
                        target_class_ids.add(cls)
                    else:
                        print(f"警告: クラスID {cls} が範囲外です（0-{CLASS_NUM-1}）。")
            
            if target_class_ids:
                print(f"\n検出対象クラス: {[cfg.dataset.class_list[cid] for cid in sorted(target_class_ids)]}")
            else:
                print("警告: 有効なクラスが指定されていません。全クラスを検出します。")
                target_class_ids = None

    # 推論フォルダ内の全画像を取得
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(INFERENCE_FOLDER, ext)))
    
    print(f"Found {len(image_files)} images in {INFERENCE_FOLDER}")
    
    # スライディングウィンドウ情報を表示
    if USE_SLIDING_WINDOW and SLIDE > 1:
        print(f"\n🔍 スライディングウィンドウモード: {SLIDE}x{SLIDE} = {SLIDE*SLIDE}タイル + 元画像")
        print(f"   小物体検出精度が向上しますが、処理時間が増加します")
    else:
        print(f"\n⚡ 通常モード（スライディングウィンドウなし）")
    
    # 推論結果の統計情報
    total_detections = 0
    class_counts = {i: 0 for i in range(CLASS_NUM)}
    detection_results = []
    images_with_detections = []  # 検出があった画像のパスを保存
    
    # 各画像に対して推論を実行
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # 画像を読み込み
            pil_image = Image.open(image_path)
            image, bbox, rev_tensor = transform(pil_image)

            image = image.to(device)[None]
            rev_tensor = rev_tensor.to(device)[None]

            with torch.no_grad():
                if USE_SLIDING_WINDOW and SLIDE > 1:
                    # スライディングウィンドウモード
                    total_image, total_shift = slide_image(image, slide=SLIDE, device=device)
                    
                    # 全タイルで推論
                    predict = model(total_image)
                    
                    # 結果を処理
                    pred_class, _, pred_bbox_raw = converter(predict["Main"])
                    
                    # オフセットを適用（元画像以外のタイル）
                    pred_bbox_raw[1:] = (pred_bbox_raw[1:] + total_shift[:, None]) / SLIDE
                    
                    # バッチ次元を統合
                    pred_bbox_raw = pred_bbox_raw.view(1, -1, 4)
                    pred_class = pred_class.view(1, -1, CLASS_NUM)
                    
                    # 座標を元の画像サイズに変換
                    pred_bbox_raw = (pred_bbox_raw - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]
                    
                    # NMSで重複を除去
                    nms_config = NMSConfig(
                        min_confidence=cfg.task.nms.min_confidence, 
                        min_iou=cfg.task.nms.min_iou,
                        max_bbox=cfg.task.nms.max_bbox
                    )
                    detections = bbox_nms(pred_class, pred_bbox_raw, nms_config)[0]
                    
                else:
                    # 通常モード
                    predict = model(image) 
                    pred_bbox = post_proccess(predict, rev_tensor)
                    detections = pred_bbox[0] if isinstance(pred_bbox, list) else pred_bbox
            
            # 特定のクラスのみをフィルタリング
            if target_class_ids is not None and len(detections) > 0:
                filtered_detections = []
                for detection in detections:
                    if len(detection) >= 6:
                        class_id = int(detection[0].item() if hasattr(detection[0], 'item') else detection[0])
                        if class_id in target_class_ids:
                            filtered_detections.append(detection)
                
                # フィルタリング結果をテンソルに変換（空の場合は空のテンソルを作成）
                if len(filtered_detections) > 0:
                    detections = torch.stack(filtered_detections)
                else:
                    # 検出がない場合は空のテンソルを作成
                    detections = torch.empty((0, 6), device=device)
            
            # クラス間NMS：異なるクラスで重複している場合、信頼度が高い方を残す
            if USE_CROSS_CLASS_NMS and len(detections) > 0:
                before_count = len(detections)
                detections = cross_class_nms(detections, iou_threshold=CROSS_CLASS_IOU_THRESHOLD)
                after_count = len(detections)
                if before_count != after_count and DEBUG_MODE:
                    print(f"\n  🔄 クラス間NMS: {before_count}個 → {after_count}個 ({before_count - after_count}個削除)")

            # デバッグモード：NMS前の生のスコアを確認
            if DEBUG_MODE and 'pred_class' in locals():
                print(f"\n  🔍 デバッグ情報 [{os.path.basename(image_path)}]:")
                print(f"     生のクラス予測形状: {pred_class.shape}")
                # 各クラスの最大スコアを表示
                max_scores = pred_class.max(dim=1)[0]  # 各予測の最大スコア
                print(f"     最大信頼度: {max_scores.max().item():.4f}")
                print(f"     平均信頼度: {max_scores.mean().item():.4f}")
                # クラスごとの最大スコア
                class_max_scores = pred_class.max(dim=0)[0]  # 各クラスの最大スコア
                for cls_id in range(CLASS_NUM):
                    score = class_max_scores[cls_id].item()
                    if score > 0.01:
                        print(f"     Class {cls_id}: 最大スコア={score:.4f}")
            
            # 検出結果の統計
            image_detections = len(detections)
            total_detections += image_detections
            
            # デバッグ：NMS後の検出数
            if DEBUG_MODE:
                print(f"     NMS後の検出数: {image_detections}個")
            
            # クラス別カウント
            image_class_counts = {i: 0 for i in range(CLASS_NUM)}
            for detection in detections:
                if len(detection) >= 6:  # [class, x1, y1, x2, y2, conf]
                    try:
                        class_id = int(detection[0].item() if hasattr(detection[0], 'item') else detection[0])  # クラスID
                        confidence = float(detection[5].item() if hasattr(detection[5], 'item') else detection[5])  # 信頼度
                        if 0 <= class_id < CLASS_NUM:
                            class_counts[class_id] += 1
                            image_class_counts[class_id] += 1
                            if DEBUG_MODE:
                                print(f"       → Class {class_id}: 信頼度={confidence:.4f}")
                    except (ValueError, IndexError, TypeError) as e:
                        print(f"    Warning: Invalid detection format: {e}")
                        continue
            
            # 結果を記録
            result_info = {
                'image': os.path.basename(image_path),
                'detections': image_detections,
                'class_counts': image_class_counts,
                'detections_detail': []
            }
            
            # 検出詳細を記録
            for detection in detections:
                if len(detection) >= 6:  # [class, x1, y1, x2, y2, conf]
                    try:
                        class_id = int(detection[0].item() if hasattr(detection[0], 'item') else detection[0])
                        confidence = float(detection[5].item() if hasattr(detection[5], 'item') else detection[5])
                        if 0 <= class_id < CLASS_NUM:
                            class_name = cfg.dataset.class_list[class_id]
                            result_info['detections_detail'].append({
                                'class': class_name,
                                'confidence': confidence,
                                'bbox': detection[1:5].tolist() if hasattr(detection[1:5], 'tolist') else detection[1:5]
                            })
                    except (ValueError, IndexError, TypeError) as e:
                        print(f"    Warning: Invalid detection format in detail recording: {e}")
                        continue
            
            detection_results.append(result_info)
            
            # 結果画像を作成
            output_image = draw_bboxes(pil_image, detections, idx2label=cfg.dataset.class_list)

            # 出力ファイル名を生成
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_result.jpg")

            # 結果を保存
            output_image.save(output_path)
            print(f"  -> Saved: {output_path}")
            print(f"  -> Detections: {image_detections}")
            if image_detections > 0:
                images_with_detections.append(output_path)  # 検出があった画像を記録
                for detail in result_info['detections_detail']:
                    print(f"    - {detail['class']}: {detail['confidence']:.3f}")
            
        except Exception as e:
            print(f"  -> Error processing {image_path}: {str(e)}")
            continue

    # 最終統計レポート
    print(f"\n" + "="*60)
    print(f"推論完了！結果は {OUTPUT_FOLDER} に保存されました。")
    print(f"="*60)
    print(f"処理した画像数: {len(image_files)}")
    print(f"総検出数: {total_detections}")
    print(f"平均検出数/画像: {total_detections/len(image_files):.2f}")
    
    print(f"\n=== クラス別検出統計 ===")
    for class_id, count in class_counts.items():
        if count > 0:
            class_name = cfg.dataset.class_list[class_id]
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"{class_name}: {count}個 ({percentage:.1f}%)")
    
    print(f"\n=== 画像別検出結果 ===")
    for result in detection_results:
        if result['detections'] > 0:
            print(f"\n{result['image']}: {result['detections']}個検出")
            for detail in result['detections_detail']:
                print(f"  - {detail['class']}: {detail['confidence']:.3f}")
        else:
            print(f"{result['image']}: 検出なし")
    
    # 検出結果をCSVファイルとして保存
    import csv
    csv_path = os.path.join(OUTPUT_FOLDER, 'detection_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['画像名', '検出数', 'クラス', '信頼度', 'バウンディングボックス'])
        
        for result in detection_results:
            if result['detections'] > 0:
                for detail in result['detections_detail']:
                    writer.writerow([
                        result['image'],
                        result['detections'],
                        detail['class'],
                        f"{detail['confidence']:.3f}",
                        str(detail['bbox'])
                    ])
            else:
                writer.writerow([result['image'], 0, 'なし', '0.000', '[]'])
    
    print(f"\n検出結果の詳細は {csv_path} に保存されました。")
    
    # 検出があった画像のみをPDFにまとめる
    if len(images_with_detections) > 0:
        pdf_path = os.path.join(OUTPUT_FOLDER, 'detections_report.pdf')
        print(f"\n{'='*60}")
        print(f"PDFレポートを作成中...")
        print(f"{'='*60}")
        
        try:
            # 最初の画像を開いてPDFとして保存
            images_to_pdf = []
            for img_path in images_with_detections:
                img = Image.open(img_path)
                # RGBモードに変換（PDFではRGBが必要）
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images_to_pdf.append(img)
            
            # 最初の画像を基準にPDFを作成し、残りを追加
            if len(images_to_pdf) > 0:
                images_to_pdf[0].save(
                    pdf_path,
                    save_all=True,
                    append_images=images_to_pdf[1:] if len(images_to_pdf) > 1 else [],
                    resolution=100.0,
                    quality=95
                )
                print(f"✅ PDFレポートを作成しました: {pdf_path}")
                print(f"   - 含まれる画像数: {len(images_with_detections)}枚")
            
        except Exception as e:
            print(f"⚠️  PDF作成中にエラーが発生しました: {str(e)}")
    else:
        print(f"\n{'='*60}")
        print(f"検出結果がないため、PDFは作成されませんでした。")
        print(f"{'='*60}")


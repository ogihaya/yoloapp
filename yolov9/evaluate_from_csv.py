"""
CSV-based Test Dataset Evaluation Script
runs/inference/results/detection_results.csvを使って精度を評価

計算する指標:
- 正解率: 正解シンボル数 / 存在シンボル数 (目標: 95%)
- 誤認識率: 誤認識シンボル数 / 存在シンボル数
- 認識漏れ割合: 認識漏れシンボル数 / 存在シンボル数
- 過検出割合: 認識しないでいいものを認識した数 / 存在シンボル数
"""

import csv
import json
import ast
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_yolo_labels(label_path: Path) -> List[Dict]:
    """
    YOLOフォーマットのラベルファイルを読み込む

    Args:
        label_path: ラベルファイルのパス

    Returns:
        ラベルのリスト [{'class_id': int, 'bbox_normalized': [x_center, y_center, w, h]}, ...]
    """
    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            # YOLO format: class_id, x_center, y_center, width, height (normalized 0-1)
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            labels.append({
                'class_id': class_id,
                'bbox_normalized': [x_center, y_center, width, height]
            })

    return labels


def denormalize_bbox(bbox_normalized: List[float], img_width: int, img_height: int) -> List[float]:
    """
    正規化されたYOLO bboxを実際の座標に変換

    Args:
        bbox_normalized: [x_center, y_center, w, h] (0-1)
        img_width: 画像の幅
        img_height: 画像の高さ

    Returns:
        [x1, y1, x2, y2]
    """
    x_center, y_center, width, height = bbox_normalized
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    return [x1, y1, x2, y2]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    2つのバウンディングボックスのIoUを計算

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU値
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_predictions_to_ground_truth(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple], List[int], List[int]]:
    """
    予測と正解ラベルをマッチング

    Args:
        predictions: 予測結果のリスト [{'class_name': str, 'bbox': [x1,y1,x2,y2], 'confidence': float}, ...]
        ground_truths: 正解ラベルのリスト [{'class_id': int, 'bbox': [x1,y1,x2,y2]}, ...]
        iou_threshold: マッチングに使用するIoU閾値

    Returns:
        matches: マッチした (pred_idx, gt_idx, iou, class_match) のリスト
        unmatched_preds: マッチしなかった予測のインデックス
        unmatched_gts: マッチしなかった正解のインデックス
    """
    matches = []
    matched_preds = set()
    matched_gts = set()

    # 全ての予測と正解の組み合わせでIoUを計算
    for pred_idx, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gts:
                continue

            iou = calculate_iou(pred['bbox'], gt['bbox'])

            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            # クラスが一致するかチェック
            class_match = (pred['class_name'] == ground_truths[best_gt_idx]['class_name'])
            matches.append((pred_idx, best_gt_idx, best_iou, class_match))
            matched_preds.add(pred_idx)
            matched_gts.add(best_gt_idx)

    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(ground_truths)) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


def evaluate_from_csv(
    detection_csv_path: str = "runs/inference/results/detection_results.csv",
    labels_dir: str = "dataset/labels/test",
    images_dir: str = "dataset/images/test",
    class_list_path: str = "yolo/config/dataset/dataset.yaml",
    iou_threshold: float = 0.5,
    output_dir: str = "runs/csv_evaluation"
):
    """
    detection_results.csvから評価を実行

    Args:
        detection_csv_path: inference.pyで生成されたdetection_results.csvのパス
        labels_dir: 正解ラベルのディレクトリ
        images_dir: 画像のディレクトリ（画像サイズ取得用）
        class_list_path: クラスリストが含まれるdataset.yamlのパス
        iou_threshold: マッチング判定のIoU閾値
        output_dir: 結果の出力ディレクトリ
    """

    from PIL import Image

    print("="*80)
    print("📊 CSV-based Evaluation")
    print("="*80)
    print(f"Detection CSV: {detection_csv_path}")
    print(f"Labels dir: {labels_dir}")
    print(f"Images dir: {images_dir}")
    print(f"IoU threshold: {iou_threshold}")
    print("="*80)

    # クラスリストを読み込み
    import yaml
    with open(class_list_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    class_list = dataset_config['class_list']

    # クラス名からIDへのマッピングを作成
    class_name_to_id = {name: i for i, name in enumerate(class_list)}

    print(f"\nクラス数: {len(class_list)}")
    print(f"クラスリスト: {class_list[:5]}... (showing first 5)")

    # detection_results.csvを読み込む
    detection_csv = Path(detection_csv_path)
    labels_path = Path(labels_dir)

    if not detection_csv.exists():
        print(f"\n❌ Error: {detection_csv_path} not found!")
        print("Please run inference.py first to generate detection_results.csv")
        return

    # CSVから予測結果を読み込み（画像ごとにグループ化）
    predictions_by_image = defaultdict(list)

    with open(detection_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row['画像名']
            class_name = row['クラス']
            confidence = float(row['信頼度'])
            bbox_str = row['バウンディングボックス']

            # バウンディングボックスをパース
            try:
                bbox = ast.literal_eval(bbox_str)
            except:
                continue

            predictions_by_image[img_name].append({
                'class_name': class_name,
                'bbox': bbox,
                'confidence': confidence
            })

    print(f"\n検出結果を読み込みました: {len(predictions_by_image)} 画像")

    # 独自指標の集計用変数
    total_ground_truths = 0  # 存在シンボル数
    total_correct = 0  # 正解シンボル数
    total_wrong_class = 0  # 誤認識シンボル数（位置は合っているがクラスが違う）
    total_false_positives = 0  # 過検出（認識しないでいいものを認識した数）
    total_false_negatives = 0  # 認識漏れシンボル数

    # クラス別の集計
    class_stats = defaultdict(lambda: {
        'total': 0,
        'correct': 0,
        'wrong_class': 0,
        'missed': 0,
        'false_positive': 0
    })

    # 画像ごとの詳細結果
    detailed_results = []

    # testフォルダ内の全画像を確認
    test_images = list(labels_path.glob("*.txt"))
    images_path = Path(images_dir)

    print(f"\n評価を開始...")
    print(f"正解ラベルファイル数: {len(test_images)}")

    for label_file in test_images:
        img_name = label_file.stem + ".jpg"  # .txtを.jpgに変換
        img_path = images_path / img_name

        # 画像の実際のサイズを取得
        if img_path.exists():
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        else:
            print(f"警告: 画像 {img_name} が見つかりません。デフォルトサイズ (2560x2560) を使用します。")
            img_width, img_height = 2560, 2560

        # 正解ラベルを読み込み（元の画像サイズで非正規化）
        ground_truths_normalized = load_yolo_labels(label_file)
        ground_truths = []

        for gt in ground_truths_normalized:
            bbox = denormalize_bbox(gt['bbox_normalized'], img_width, img_height)
            ground_truths.append({
                'class_id': gt['class_id'],
                'class_name': class_list[gt['class_id']],
                'bbox': bbox
            })

        # この画像の予測結果を取得
        predictions = predictions_by_image.get(img_name, [])

        # マッチング
        matches, unmatched_preds, unmatched_gts = match_predictions_to_ground_truth(
            predictions, ground_truths, iou_threshold
        )

        # 画像ごとの統計
        img_total_gt = len(ground_truths)
        img_correct = sum(1 for m in matches if m[3])  # class_matchがTrue
        img_wrong_class = sum(1 for m in matches if not m[3])  # class_matchがFalse
        img_false_positive = len(unmatched_preds)
        img_false_negative = len(unmatched_gts)

        # 全体の集計に追加
        total_ground_truths += img_total_gt
        total_correct += img_correct
        total_wrong_class += img_wrong_class
        total_false_positives += img_false_positive
        total_false_negatives += img_false_negative

        # クラス別の集計
        for gt in ground_truths:
            class_stats[gt['class_name']]['total'] += 1

        for match in matches:
            pred_idx, gt_idx, iou, class_match = match
            gt_class = ground_truths[gt_idx]['class_name']
            if class_match:
                class_stats[gt_class]['correct'] += 1
            else:
                class_stats[gt_class]['wrong_class'] += 1

        for gt_idx in unmatched_gts:
            gt_class = ground_truths[gt_idx]['class_name']
            class_stats[gt_class]['missed'] += 1

        for pred_idx in unmatched_preds:
            pred_class = predictions[pred_idx]['class_name']
            class_stats[pred_class]['false_positive'] += 1

        # 詳細結果を記録
        detailed_results.append({
            'image': img_name,
            'total_gt': img_total_gt,
            'correct': img_correct,
            'wrong_class': img_wrong_class,
            'false_positive': img_false_positive,
            'false_negative': img_false_negative
        })

    # 独自指標を計算
    print(f"\n📈 計算中...")

    if total_ground_truths > 0:
        accuracy_rate = (total_correct / total_ground_truths) * 100
        wrong_class_rate = (total_wrong_class / total_ground_truths) * 100
        miss_rate = (total_false_negatives / total_ground_truths) * 100
        over_detection_rate = (total_false_positives / total_ground_truths) * 100
    else:
        accuracy_rate = wrong_class_rate = miss_rate = over_detection_rate = 0.0

    # 結果を表示
    print("\n" + "="*80)
    print("📊 CUSTOM EVALUATION METRICS")
    print("="*80)
    print(f"\n存在シンボル総数: {total_ground_truths}")
    print(f"\n🎯 精度指標:")
    print(f"   ✅ 正解率:              {accuracy_rate:.2f}% ({total_correct}/{total_ground_truths}) {'🎯 目標達成!' if accuracy_rate >= 95 else '⚠️  目標未達 (目標: 95%)'}")
    print(f"   ❌ 誤認識率:            {wrong_class_rate:.2f}% ({total_wrong_class}/{total_ground_truths})")
    print(f"   ⚠️  認識漏れ割合:       {miss_rate:.2f}% ({total_false_negatives}/{total_ground_truths})")
    print(f"   🔺 過検出割合:          {over_detection_rate:.2f}% ({total_false_positives}/{total_ground_truths})")

    # クラス別の統計
    print(f"\n📋 クラス別統計:")
    class_results = []
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        total = stats['total']
        if total > 0:
            class_acc = (stats['correct'] / total) * 100
        else:
            class_acc = 0.0

        print(f"   {class_name}:")
        print(f"      Total: {total}, Correct: {stats['correct']}, "
              f"Wrong class: {stats['wrong_class']}, Missed: {stats['missed']}, "
              f"False positive: {stats['false_positive']}, Accuracy: {class_acc:.2f}%")

        class_results.append({
            'class_name': class_name,
            'total': total,
            'correct': stats['correct'],
            'wrong_class': stats['wrong_class'],
            'missed': stats['missed'],
            'false_positive': stats['false_positive'],
            'accuracy': class_acc
        })

    print("="*80)

    # 結果を保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON形式で詳細結果を保存
    results_json = {
        "detection_csv": detection_csv_path,
        "labels_dir": labels_dir,
        "images_dir": images_dir,
        "evaluation_settings": {
            "iou_threshold": iou_threshold,
            "note": "Image sizes are read from actual image files"
        },
        "custom_metrics": {
            "total_ground_truths": total_ground_truths,
            "total_correct": total_correct,
            "total_wrong_class": total_wrong_class,
            "total_false_positives": total_false_positives,
            "total_false_negatives": total_false_negatives,
            "accuracy_rate": accuracy_rate,
            "wrong_class_rate": wrong_class_rate,
            "miss_rate": miss_rate,
            "over_detection_rate": over_detection_rate,
            "target_achieved": accuracy_rate >= 95
        },
        "class_statistics": class_results
    }

    json_path = output_path / "evaluation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {json_path}")

    # CSV形式でクラス別統計を保存
    csv_path = output_path / "class_statistics.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'クラス名', '存在数', '正解数', '誤認識数',
            '認識漏れ数', '過検出数', '正解率(%)'
        ])
        for cr in class_results:
            writer.writerow([
                cr['class_name'],
                cr['total'],
                cr['correct'],
                cr['wrong_class'],
                cr['missed'],
                cr['false_positive'],
                f"{cr['accuracy']:.2f}"
            ])
    print(f"💾 Class statistics saved to: {csv_path}")

    # 画像ごとの詳細結果をCSVで保存
    details_csv_path = output_path / "image_details.csv"
    with open(details_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '画像名', '正解総数', '正解数', '誤認識数', '過検出数', '認識漏れ数', '正解率(%)'
        ])
        for result in detailed_results:
            img_acc = (result['correct'] / result['total_gt'] * 100) if result['total_gt'] > 0 else 0
            writer.writerow([
                result['image'],
                result['total_gt'],
                result['correct'],
                result['wrong_class'],
                result['false_positive'],
                result['false_negative'],
                f"{img_acc:.2f}"
            ])
    print(f"💾 Image details saved to: {details_csv_path}")

    return results_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate using detection_results.csv")
    parser.add_argument(
        "--detection-csv",
        type=str,
        default="runs/inference/results/detection_results.csv",
        help="Path to detection_results.csv"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="dataset/labels/test",
        help="Directory containing ground truth labels"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="dataset/images/test",
        help="Directory containing test images (for reading actual image size)"
    )
    parser.add_argument(
        "--class-list",
        type=str,
        default="yolo/config/dataset/dataset.yaml",
        help="Path to dataset.yaml containing class list"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/csv_evaluation",
        help="Output directory"
    )

    args = parser.parse_args()

    results = evaluate_from_csv(
        detection_csv_path=args.detection_csv,
        labels_dir=args.labels_dir,
        images_dir=args.images_dir,
        class_list_path=args.class_list,
        iou_threshold=args.iou_threshold,
        output_dir=args.output_dir
    )

    print("\n✅ Evaluation completed successfully!")

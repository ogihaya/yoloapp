# 既存リソース

- `yolov9/` フォルダー: 動作確認済みのYOLOv9実装が含まれています
  - このフォルダー内の不要なファイルやテストファイルを整理してください
  - **推論の実行手順**:
    1. 推論したい画像を `yolov9/dataset/images/inference` フォルダーに配置します
    2. `yolov9/yolo/config` フォルダー内の `general.yml` と `task/inference.yml` の設定を行います(GPUがない場合はデバイスの設定も変える必要あり)
    3. 学習済み重みファイルを `yolov9/weights/best.pt` として配置します
    4. `python yolo/inference.py` を実行します
    5. 推論結果の画像（検出されたクラス名と信頼度（%）が表示された画像）が `yolov9/runs/inference/results/` フォルダーに出力されます
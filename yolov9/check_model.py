import torch
import sys
from pathlib import Path

def check_model_info(weight_path):
    """
    モデルファイルの詳細情報を表示
    """
    print("="*70)
    print(f"📦 モデルファイル情報: {weight_path}")
    print("="*70)
    
    try:
        # モデルをロード
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        # チェックポイントの構造を確認
        if isinstance(checkpoint, dict):
            print(f"\n📋 チェックポイントの構造:")
            print(f"  キー: {list(checkpoint.keys())}")
            
            # state_dictを取得
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
                print(f"\n  ✅ 'model' キーが存在します")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"\n  ✅ 'state_dict' キーが存在します")
            else:
                # チェックポイント自体がstate_dict
                state_dict = checkpoint
                print(f"\n  ℹ️  チェックポイント自体がstate_dictです")
        else:
            state_dict = checkpoint
            print(f"\n  ℹ️  チェックポイントは直接的なstate_dictです")
        
        print(f"\n📊 パラメータ総数: {len(state_dict)} 個")
        
        # クラス数を推定
        print(f"\n🔍 クラス数の検出:")
        class_num = None
        
        # クラス分類層のパラメータを探す
        for key, value in state_dict.items():
            if 'class_conv' in key and 'weight' in key and value.dim() == 4:
                shape = value.shape
                detected_class_num = shape[0]  # 最初の次元がクラス数
                print(f"  {key}")
                print(f"    形状: {shape}")
                print(f"    → クラス数: {detected_class_num}")
                
                if class_num is None:
                    class_num = detected_class_num
                elif class_num != detected_class_num:
                    print(f"    ⚠️  警告: 異なるクラス数が検出されました")
        
        if class_num:
            print(f"\n✅ 推定クラス数: {class_num} クラス")
        else:
            print(f"\n⚠️  クラス数を検出できませんでした")
        
        # 主要なレイヤーの形状を表示
        print(f"\n📐 主要なレイヤーの形状:")
        important_keys = []
        for key in state_dict.keys():
            if any(pattern in key for pattern in ['class_conv.2.weight', 'class_conv.2.bias', 
                                                    'anchor_conv', 'conv.weight']):
                important_keys.append(key)
        
        # 最初の10個だけ表示
        for key in sorted(important_keys)[:10]:
            value = state_dict[key]
            print(f"  {key}: {value.shape}")
        
        if len(important_keys) > 10:
            print(f"  ... 他 {len(important_keys) - 10} 個")
        
        # モデルサイズ
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"\n💾 総パラメータ数: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # ファイルサイズ
        file_size = Path(weight_path).stat().st_size
        print(f"📁 ファイルサイズ: {file_size / (1024**2):.2f} MB")
        
        print("\n" + "="*70)
        
        return class_num
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # デフォルトのパス
    default_weights = [
        "weights/best.pt",
        "weights/best_ver1.pt",
        "weights/best_ver3.pt",
        "weights/v9-s.pt",
    ]
    
    if len(sys.argv) > 1:
        # コマンドライン引数で指定
        weight_path = sys.argv[1]
        check_model_info(weight_path)
    else:
        # 全てのデフォルトweightsをチェック
        print("\n🔍 全てのモデルファイルをチェックします...\n")
        
        for weight_path in default_weights:
            if Path(weight_path).exists():
                check_model_info(weight_path)
                print("\n")
            else:
                print(f"⏭️  {weight_path} は存在しません\n")


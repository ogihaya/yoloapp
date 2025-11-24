# 環境変数の設定方法

このプロジェクトでは、セキュリティ上の理由から、機密情報（特にDjangoの`SECRET_KEY`）を環境変数で管理します。

## セットアップ手順

### 1. `.env`ファイルの作成

プロジェクトのルートディレクトリ（`manage.py`がある`yoloApp2`フォルダと同じ階層）に、`.env`ファイルを作成してください。

```bash
# Windows PowerShellの場合
Copy-Item .env.example .env

# Linux/Macの場合
cp .env.example .env
```

### 2. `.env`ファイルの編集

`.env`ファイルを開き、以下の値を設定してください：

```env
# Djangoの秘密鍵（必須）
# 本番環境では、必ず強力なランダムな文字列に変更してください
DJANGO_SECRET_KEY=あなたの秘密鍵をここに設定
```

### 3. 秘密鍵の生成方法

新しい秘密鍵を生成するには、以下のコマンドを実行してください：

```bash
# Djangoがインストールされている環境で実行
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

生成された文字列を`.env`ファイルの`DJANGO_SECRET_KEY`に設定してください。

## 注意事項

- **`.env`ファイルは絶対にGitHubにアップロードしないでください**
  - `.env`ファイルは`.gitignore`に含まれているため、通常は自動的に除外されます
  - 誤ってコミットしないよう、注意してください

- **本番環境では必ず強力な秘密鍵を使用してください**
  - 開発用のデフォルト値は、セキュリティ上脆弱です
  - 本番環境では、必ず新しい強力な秘密鍵を生成して設定してください

## 環境変数の読み込み

このプロジェクトでは、標準ライブラリの`os.environ.get()`を使用して環境変数を読み込んでいます。

より高度な機能（`.env`ファイルの自動読み込みなど）が必要な場合は、`python-dotenv`パッケージの使用を検討してください。


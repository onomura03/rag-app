# Pythonの公式スリムイメージを使用 (軽量版)
FROM python:3.9-slim

# 作業ディレクトリを /app に設定
WORKDIR /app

# 依存関係ファイルをコンテナにコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Streamlitがシークレットを読み込めるように、secrets.tomlを正しいパスにコピー
# ローカルの .streamlit/secrets.toml をコンテナの .streamlit/ にコピー
COPY .streamlit/secrets.toml ./.streamlit/secrets.toml

# その他のアプリケーションファイルをコンテナにコピー
# .streamlitディレクトリもコピー対象に含める
COPY . .

# アプリケーションのポートを公開
EXPOSE 8501

# Streamlitアプリケーションを起動
# 実行するファイルが 'rag_app.py' であれば、こちらを使用
# CMD ["streamlit", "run", "rag_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# 実行するファイルが 'app.py' であれば、こちらを使用
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

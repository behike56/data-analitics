# syntax=docker/dockerfile:1
FROM python:3.12.11-slim-bookworm

# ----- 0) Poetry 本体をインストール -----
ARG POETRY_VERSION=2.1.3
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --no-cache-dir "poetry==$POETRY_VERSION" && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ----- 1) 作業ディレクトリ -----
WORKDIR /app

# ----- 2) Poetry の設定 -----
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1

# キャッシュ効率のためまず定義ファイルだけコピー
COPY pyproject.toml poetry.lock* /app/

# ----- 3) 依存インストール -----
RUN poetry install --no-root --no-interaction --no-ansi

# Jupyter 本体＋ipykernel だけ別途入れる
RUN pip install --no-cache-dir jupyterlab ipykernel

# ----- 4) プロジェクトのソースコード -----
COPY . /app

# ----- 5) カーネル登録（表示名はお好みで） -----
RUN python -m ipykernel install --user \
        --name "poetry" \
        --display-name "Python (poetry)"

# ----- 6) Jupyter Lab を公開 -----
EXPOSE 8888
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''", \
     "--allow-root"]

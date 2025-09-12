FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    htop \
    nvtop \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ollamaのインストール
RUN curl -fsSL https://ollama.com/install.sh | sh

# 作業ディレクトリの設定
WORKDIR /workspace

# Python依存関係のインストール
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Unslothの最新版インストール
RUN pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip3 install --upgrade xformers

# Flash Attention 2のインストール
RUN pip3 install flash-attn --no-build-isolation

# スクリプトとディレクトリのコピー
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY models/ ./models/
COPY notebooks/ ./notebooks/

# Ollamaサービス起動スクリプト
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
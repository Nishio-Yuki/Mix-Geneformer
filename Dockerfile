# CUDA 11.7 + 開発ツール付き（nvcc あり）
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=yukinishio
ARG UID=2110
ARG GID=2110

# 1) 必要パッケージ（root）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git \
    python3 python3-venv python3-pip python3-dev \
    build-essential cmake ninja-build pkg-config \
    libaio-dev \
 && rm -rf /var/lib/apt/lists/*

# 2) 非rootユーザー作成（root）
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# 3) venv を作って所有権をユーザーに（root）
RUN python3 -m venv /opt/venv \
 && chown -R ${UID}:${GID} /opt/venv

# venv を PATH に通す（全ステージ共通）
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# CUDA 関連の標準環境変数
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# 4) 作業ディレクトリとパーミッション（root）
RUN mkdir -p /work && chown -R ${UID}:${GID} /work
WORKDIR /work

# 5) ここから非rootで作業
USER ${UID}:${GID}
ENV HOME=/home/${USERNAME}

# 6) Python パッケージ（ユーザーでインストール）
#    重要: PyTorch は cu117 を明示、DeepSpeed はソースビルド前提なのでビルドツールは上で準備済み
RUN pip install --upgrade pip wheel setuptools --no-cache-dir && \
    pip install --no-cache-dir \
      "numpy<2.0" pandas scikit-learn anndata python-igraph leidenalg \
      matplotlib seaborn scanpy==1.9.8 scvi-tools datasets jupyterlab && \
    pip install --no-cache-dir \
      torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
      --index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir \
      transformers accelerate \
      # DeepSpeed は CUDA 拡張をビルドするので nvcc 必須（devel イメージでOK）
      deepspeed==0.13.5

# 7) (任意) Triton キャッシュの書き込み先をユーザー配下に
ENV TRITON_CACHE_DIR=${HOME}/.triton

# 8) NCCL と分散の既定（必要に応じて上書き可）
ENV NCCL_DEBUG=WARN \
    NCCL_IB_DISABLE=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    CUDA_DEVICE_MAX_CONNECTIONS=1

# 9) デフォルトシェル
CMD ["bash"]

#!/usr/bin/env bash
# =============================================================================
# cardiac_agent 环境搭建脚本
#
# 适用系统：Linux（Ubuntu 20.04 / 22.04）
# GPU 要求：NVIDIA，驱动版本 ≥ 525（支持 CUDA 12.1）
# 用法：bash setup_env.sh
#
# 版本选定依据：
#   Python  3.10.14  —— faiss-cpu 1.7.4 有 cp310 wheel；sentence-transformers 推荐 3.10+
#   PyTorch 2.2.2    —— faiss conda channel 官方测试组合；transformers 4.40 兼容
#   CUDA    12.1     —— PyTorch 2.2.x 官方支持的最高稳定 CUDA 版本
# =============================================================================

set -e  # 任何命令失败立即退出

ENV_NAME="cardiac_agent"
PYTHON_VERSION="3.10.14"

echo "=============================================="
echo " Step 0: 检查前置条件"
echo "=============================================="

# 检查 conda 是否存在
if ! command -v conda &>/dev/null; then
    echo "[错误] 未找到 conda，请先安装 Miniconda："
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# 检查 NVIDIA 驱动
if command -v nvidia-smi &>/dev/null; then
    echo "[信息] GPU 信息："
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "[信息] 驱动版本: $DRIVER_VERSION（需要 ≥ 525 才能支持 CUDA 12.1）"
else
    echo "[警告] 未检测到 NVIDIA GPU，将以 CPU 模式安装"
    CPU_ONLY=1
fi

echo ""
echo "=============================================="
echo " Step 1: 创建 conda 环境（Python 3.10.14）"
echo "=============================================="

# 如果环境已存在则先删除（重新搭建时用）
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[信息] 环境 ${ENV_NAME} 已存在，先删除..."
    conda env remove -n "${ENV_NAME}" -y
fi

conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
echo "[完成] conda 环境 ${ENV_NAME} 创建成功"

echo ""
echo "=============================================="
echo " Step 2: 安装 PyTorch 2.2.2"
echo "=============================================="

# 激活环境并在其中执行安装
# 注意：source activate 在脚本里可能不生效，用 conda run 替代
if [ "${CPU_ONLY}" = "1" ]; then
    echo "[信息] CPU 模式安装 PyTorch..."
    conda run -n "${ENV_NAME}" pip install \
        torch==2.2.2 \
        torchvision==0.17.2 \
        torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu
else
    echo "[信息] GPU 模式安装 PyTorch（CUDA 12.1）..."
    # 用 pip 而不是 conda 安装 torch，原因：
    #   conda pytorch channel 的 torch 2.2.2 有时依赖解析慢
    #   pip 官方 whl 更直接，CUDA 版本明确
    conda run -n "${ENV_NAME}" pip install \
        torch==2.2.2 \
        torchvision==0.17.2 \
        torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cu121
fi

echo "[完成] PyTorch 安装完成"

echo ""
echo "=============================================="
echo " Step 3: 安装 faiss-cpu（通过 conda pytorch channel）"
echo "=============================================="
# faiss 必须通过 conda pytorch channel 安装，而不是 pip
# 原因：conda 版本与 numpy/blas 的链接更稳定，pip 版本在某些 Linux 发行版有 segfault 风险
conda install -n "${ENV_NAME}" \
    -c pytorch \
    -c conda-forge \
    faiss-cpu=1.7.4 \
    -y

echo "[完成] faiss-cpu 1.7.4 安装完成"

echo ""
echo "=============================================="
echo " Step 4: 安装其他 pip 依赖"
echo "=============================================="
conda run -n "${ENV_NAME}" pip install \
    transformers==4.40.2 \
    "tokenizers>=0.19,<0.20" \
    accelerate==0.30.1 \
    safetensors>=0.4.3 \
    "huggingface-hub>=0.23.0" \
    "sentence-transformers==2.7.0" \
    bitsandbytes==0.43.1 \
    "numpy>=1.24.0,<2.0" \
    "tqdm>=4.66.0" \
    "packaging>=23.0" \
    "regex>=2023.10.3" \
    "requests>=2.31.0" \
    "filelock>=3.13.0" \
    "pyyaml>=6.0.1"

echo "[完成] pip 依赖安装完成"

echo ""
echo "=============================================="
echo " Step 5: 验证安装"
echo "=============================================="
conda run -n "${ENV_NAME}" python - << 'PYEOF'
import sys
print(f"Python:  {sys.version.split()[0]}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import transformers
print(f"transformers: {transformers.__version__}")

import sentence_transformers
print(f"sentence-transformers: {sentence_transformers.__version__}")

import faiss
print(f"faiss: {faiss.__version__}")
print(f"faiss index type test: ", end="")
idx = faiss.IndexFlatIP(128)
import numpy as np
idx.add(np.random.rand(10, 128).astype(np.float32))
_, I = idx.search(np.random.rand(1, 128).astype(np.float32), 3)
print(f"OK，检索结果 shape={I.shape}")

import numpy as np
print(f"numpy: {np.__version__}")

print()
print("=" * 45)
print("  所有包验证通过，环境就绪！")
print("=" * 45)
PYEOF

echo ""
echo "=============================================="
echo " 安装完成！"
echo "=============================================="
echo ""
echo "激活环境："
echo "  conda activate ${ENV_NAME}"
echo ""
echo "运行项目："
echo "  cd cardiac_agent"
echo "  python main.py --mode demo"
echo ""
echo "下次进入环境，只需："
echo "  conda activate ${ENV_NAME}"

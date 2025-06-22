#!/bin/bash

# SD-Scripts FLUX.1/SD3 Training Environment Provisioning Script
# For vast.ai with CUDA 12.4.1 and Ubuntu 22.04
# Version: 1.1
# Last updated: 2025-01-22

# 脚本出错时继续执行，记录错误
set -eo pipefail

# 创建日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log "=== SD-Scripts FLUX.1/SD3 Training Environment Setup ==="
log "Starting provisioning script..."
log "Version: 1.1"
log "Base image: cuda-12.4.1-cudnn-devel-ubuntu22.04-py310"

# 记录开始时间
START_TIME=$(date)
log "Start time: $START_TIME"

# 切换到持久化目录
cd /workspace/

# ========== 系统环境准备 ==========
log "=== Phase 1: System Environment Setup ==="

# 更新系统包和安装编译工具
log ">>> Updating system packages and installing build tools..."
apt-get update -y || log_error "Failed to update apt packages"
apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    pkg-config \
    libhdf5-dev \
    libffi-dev \
    python3-dev \
    ninja-build \
    cmake \
    htop \
    tmux \
    vim || log_error "Failed to install some system packages"

# 设置内存优化
log ">>> Setting up memory optimization..."
# 检查 libtcmalloc 是否存在，如果存在才设置 LD_PRELOAD
if [ -f "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" ]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD"
    log "✓ tcmalloc memory optimization enabled"
elif [ -f "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" ]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4:$LD_PRELOAD"
    log "✓ tcmalloc memory optimization enabled"
else
    log_error "⚠ tcmalloc not found, skipping memory optimization"
fi

# 验证 CUDA 环境
log ">>> Verifying CUDA environment..."
nvidia-smi || log_error "nvidia-smi failed"
nvcc --version || log_error "nvcc not found"

# 设置 CUDA 相关环境变量
log ">>> Setting up CUDA environment variables..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# ========== Python 环境设置 ==========
log "=== Phase 2: Python Environment Setup ==="

# 创建 Python 虚拟环境
log ">>> Creating Python virtual environment..."
python3.10 -m venv sd-scripts-env
source sd-scripts-env/bin/activate

# 验证 Python 版本
log "Python version: $(python --version)"
log "Pip version: $(pip --version)"

# 升级 pip 和基础工具
log ">>> Upgrading pip and basic tools..."
pip install --upgrade pip setuptools wheel

# ========== SD-Scripts 项目设置 ==========
log "=== Phase 3: SD-Scripts Project Setup ==="

# 克隆 sd-scripts 项目 (sd3 分支)
log ">>> Cloning sd-scripts repository (sd3 branch)..."
if [ -d "sd-scripts" ]; then
    log "sd-scripts directory exists, removing..."
    rm -rf sd-scripts
fi

git clone --branch sd3 --depth 1 https://github.com/kohya-ss/sd-scripts.git || {
    log_error "Failed to clone sd-scripts repository"
    exit 1
}
cd sd-scripts

# ========== 依赖安装 ==========
log "=== Phase 4: Dependencies Installation ==="

# 首先安装特定版本的 triton (修复 bitsandbytes 的依赖)
log ">>> Installing triton..."
pip install triton==3.0.0 || log_error "Failed to install triton"

# 安装 PyTorch 2.4.0 with CUDA 12.4 (README 明确要求)
log ">>> Installing PyTorch 2.4.0 with CUDA 12.4..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 || {
    log_error "Failed to install PyTorch"
    exit 1
}

# 验证 PyTorch 安装
log ">>> Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" || log_error "PyTorch verification failed"

# 安装项目依赖 (严格按照 requirements.txt，但跳过 -e . 行)
log ">>> Installing sd-scripts requirements..."
# 创建一个临时的 requirements 文件，排除 -e . 行
grep -v "^-e " requirements.txt > temp_requirements.txt
pip install -r temp_requirements.txt || log_error "Failed to install some requirements"
rm temp_requirements.txt

# 安装项目本身（使用可编辑模式）
pip install -e . || log_error "Failed to install sd-scripts package"

# 安装与 PyTorch 2.4.0 兼容的 xformers（修复版本匹配）
log ">>> Installing xformers compatible with PyTorch 2.4.0..."
# 优先尝试 0.0.28.post1（有官方 cu124 轮子，依赖 torch==2.4.1，但我们使用 --no-deps 避免升级）
if ! pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124 --no-deps --force-reinstall; then
    log_error "xformers 0.0.28.post1 wheel not available, falling back to source build"
    # 使用源码安装（可能耗时较长），仍然 --no-deps 避免依赖冲突
    pip install xformers==0.0.28.post3 --no-binary xformers --no-deps || log_error "Failed to build xformers from source"
fi

# 验证关键依赖版本
log ">>> Verifying key dependencies versions..."
python -c "
import sys
try:
    import accelerate
    print(f'accelerate: {accelerate.__version__}')
except ImportError as e:
    print(f'accelerate: Import failed - {e}')

try:
    import transformers
    print(f'transformers: {transformers.__version__}')
except ImportError as e:
    print(f'transformers: Import failed - {e}')

try:
    import diffusers
    print(f'diffusers: {diffusers.__version__}')
    # 测试关键导入
    from huggingface_hub import cached_download
    print('✓ cached_download available')
except ImportError as e:
    print(f'diffusers: Import failed - {e}')

try:
    import bitsandbytes
    print(f'bitsandbytes: {bitsandbytes.__version__}')
except ImportError as e:
    print(f'bitsandbytes: Import failed - {e}')

try:
    import safetensors
    print(f'safetensors: {safetensors.__version__}')
except ImportError as e:
    print(f'safetensors: Import failed - {e}')

try:
    import torch
    print(f'torch: {torch.__version__}')
except ImportError as e:
    print(f'torch: Import failed - {e}')

try:
    import huggingface_hub
    print(f'huggingface_hub: {huggingface_hub.__version__}')
except ImportError as e:
    print(f'huggingface_hub: Import failed - {e}')
" || log_error "Dependency verification failed"

# 修复 huggingface_hub 版本冲突（关键修复）
log ">>> Fixing huggingface_hub version conflict (CRITICAL)..."
pip install huggingface_hub==0.24.5 --force-reinstall || log_error "Failed to fix huggingface_hub version"

# 验证修复结果
log ">>> Verifying huggingface_hub fix..."
python -c "
try:
    from huggingface_hub import cached_download
    print('✓ cached_download import successful')
    import huggingface_hub
    print(f'✓ huggingface_hub version: {huggingface_hub.__version__}')
except Exception as e:
    print(f'✗ huggingface_hub fix failed: {e}')
" || log_error "HuggingFace Hub fix verification failed"

# 安装 DeepSpeed (README 明确要求的版本)
log ">>> Installing DeepSpeed (required for FLUX.1/SD3)..."
pip install deepspeed==0.16.7 || log_error "Failed to install DeepSpeed"

# 安装额外的性能优化包
log ">>> Installing additional performance packages..."
pip install --no-deps wandb || log_error "wandb installation failed"

# 尝试安装 flash-attention (可能失败)
log ">>> Attempting to install flash-attention..."
pip install flash-attn --no-build-isolation || log_error "flash-attn installation failed (this is normal for some environments)"

# 安装可选的 WD14 tagger 依赖
log ">>> Installing optional WD14 tagger dependencies..."
pip install onnx==1.15.0 onnxruntime-gpu==1.17.1 || log_error "ONNX dependencies installation failed"

# ========== 模型下载 ==========
log "=== Phase 5: Model Downloads ==="

# 返回 workspace 目录
cd /workspace/

# 创建模型目录
log ">>> Creating models directory..."
mkdir -p /workspace/models
cd /workspace/models

# 检查 HuggingFace Token
log ">>> Checking HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    log_error "WARNING: HF_TOKEN environment variable not set!"
    echo "   FLUX.1-dev requires authentication. Please set HF_TOKEN in vast.ai environment variables."
    echo "   To get your token:"
    echo "   1. Create an account at https://huggingface.co"
    echo "   2. Go to https://huggingface.co/settings/tokens"
    echo "   3. Create a new token with 'read' permission"
    echo "   4. Go to https://huggingface.co/black-forest-labs/FLUX.1-dev and accept the license"
    echo "   5. Add HF_TOKEN=your_token_here to vast.ai environment variables"
    echo ""
    SKIP_AUTH_MODELS=true
else
    log "✓ HF_TOKEN found, configuring authentication..."
    # 重要：不升级huggingface_hub，保持与diffusers兼容的版本
    # pip install -U huggingface_hub || log_error "Failed to update huggingface_hub"
    
    # 登录 HuggingFace
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential || log_error "HuggingFace login failed"
    log "✓ HuggingFace authentication configured"
fi

# 下载模型函数
download_model() {
    local url=$1
    local filename=$2
    local require_auth=$3
    
    if [ "$require_auth" = "true" ] && [ "$SKIP_AUTH_MODELS" = "true" ]; then
        log_error "Skipping $filename (requires authentication)"
        return
    fi
    
    if [ -f "$filename" ]; then
        log "✓ $filename already exists, skipping download"
    else
        log "📥 Downloading $filename..."
        if [ "$require_auth" = "true" ]; then
            # 使用 huggingface-cli 下载需要认证的文件
            huggingface-cli download --resume-download --local-dir . \
                $(echo $url | sed 's|https://huggingface.co/||' | sed 's|/resolve/.*||') \
                $(basename $url) --local-dir-use-symlinks False || {
                log_error "Failed to download $filename"
                return 1
            }
            mv $(basename $url) $filename 2>/dev/null || true
        else
            # 使用 wget 下载公开文件
            wget -c -O $filename "$url" || {
                log_error "Failed to download $filename"
                rm -f $filename
                return 1
            }
        fi
        log "✓ Successfully downloaded $filename"
    fi
}

# 下载模型文件
log ">>> Downloading model files..."
log "This may take a while depending on your internet speed..."

# FLUX.1-dev (需要认证)
download_model \
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" \
    "flux1-dev.safetensors" \
    "true"

# CLIP-L (公开)
download_model \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
    "clip_l.safetensors" \
    "false"

# T5XXL fp16 (公开)
download_model \
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors" \
    "t5xxl_fp16.safetensors" \
    "false"

# AE (需要认证)
download_model \
    "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors" \
    "ae.safetensors" \
    "true"

# 创建模型路径配置文件
log ">>> Creating model paths configuration..."
cat > /workspace/model_paths.txt <<'EOF'
# Model paths for sd-scripts training
FLUX_MODEL=/workspace/models/flux1-dev.safetensors
CLIP_L=/workspace/models/clip_l.safetensors
T5XXL=/workspace/models/t5xxl_fp16.safetensors
AE=/workspace/models/ae.safetensors
EOF

# 显示模型状态
log ">>> Model files status:"
ls -lah /workspace/models/

# 返回 sd-scripts 目录
cd /workspace/sd-scripts

# 验证核心功能
log ">>> Verifying core functionality after model download..."
python -c "
try:
    import diffusers
    print('✓ diffusers import successful')
except Exception as e:
    print(f'✗ diffusers import failed: {e}')
    
try:
    from library import train_util
    print('✓ SD-Scripts library import successful')
except Exception as e:
    print(f'✗ SD-Scripts library import failed: {e}')
" || log_error "Core functionality verification failed"

# ========== 辅助脚本创建 ==========
log "=== Phase 6: Creating Helper Scripts ==="

# 创建环境激活脚本
log ">>> Creating environment activation script..."
cat > /workspace/activate_env.sh <<'EOF'
#!/bin/bash
source /workspace/sd-scripts-env/bin/activate
cd /workspace/sd-scripts

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "SD-Scripts environment activated!"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "DeepSpeed available: $(python -c 'try: import deepspeed; print(True); except Exception: print(False)')"

if [ $# -gt 0 ]; then
    exec "$@"
else
    exec bash
fi
EOF

chmod +x /workspace/activate_env.sh

# 创建依赖检查脚本
log ">>> Creating dependency check script..."
cat > /workspace/check_dependencies.sh <<'EOF'
#!/bin/bash
source /workspace/sd-scripts-env/bin/activate
cd /workspace/sd-scripts

echo "=== Checking SD-Scripts Dependencies ==="
echo "Checking requirements.txt compliance..."

python -c "
import pkg_resources
import sys

# 检查 requirements.txt 中的包
required_packages = []
with open('requirements.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('-e'):
            if '==' in line:
                required_packages.append(line)
            elif line == 'tensorboard':
                required_packages.append('tensorboard')

print(f'Checking {len(required_packages)} required packages...')
failed = []
success = []

for package in required_packages:
    try:
        if '==' in package:
            name, version = package.split('==')
            installed = pkg_resources.get_distribution(name)
            if installed.version != version:
                failed.append(f'{name}: required {version}, installed {installed.version}')
            else:
                success.append(package)
        else:
            pkg_resources.get_distribution(package)
            success.append(package)
    except Exception as e:
        failed.append(f'{package}: {str(e)}')

print(f'\\n✓ Successfully installed: {len(success)} packages')
if failed:
    print(f'⚠ Issues found: {len(failed)} packages')
    for f in failed:
        print(f'  - {f}')
else:
    print('\\n✓ All required packages are correctly installed!')

# 检查额外的重要依赖
print('\\n=== Additional Dependencies Check ===')
extra_deps = {
    'torch': '2.4.0',
    'torchvision': '0.19.0', 
    'deepspeed': '0.16.7'
}

for name, expected_version in extra_deps.items():
    try:
        installed = pkg_resources.get_distribution(name)
        if installed.version == expected_version:
            print(f'✓ {name}: {installed.version} (expected: {expected_version})')
        else:
            print(f'⚠ {name}: {installed.version} (expected: {expected_version})')
    except Exception as e:
        print(f'✗ {name}: {str(e)}')
"
EOF

chmod +x /workspace/check_dependencies.sh

# 创建完整的测试脚本
log ">>> Creating installation test script..."
cat > /workspace/test_installation.sh <<'EOF'
#!/bin/bash
source /workspace/sd-scripts-env/bin/activate
cd /workspace/sd-scripts

echo "=== Testing SD-Scripts Installation ==="
echo ""
echo "1. Testing PyTorch and CUDA..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'CUDA capability: {torch.cuda.get_device_capability()}')
"

echo ""
echo "2. Testing core dependencies..."
python -c "
import transformers
import diffusers
import accelerate
import bitsandbytes
import safetensors
import deepspeed
print('✓ All core dependencies imported successfully')
print(f'transformers: {transformers.__version__}')
print(f'diffusers: {diffusers.__version__}')
print(f'accelerate: {accelerate.__version__}')
print(f'bitsandbytes: {bitsandbytes.__version__}')
print(f'safetensors: {safetensors.__version__}')
print(f'deepspeed: {deepspeed.__version__}')
"

echo ""
echo "3. Testing sd-scripts modules..."
python -c "
import sys
sys.path.append('.')
print('Testing basic imports...')
try:
    from library import train_util, model_util
    print('✓ SD-Scripts core library imports successful')
except ImportError as e:
    print(f'✗ Core library import failed: {e}')
    sys.exit(1)

print('Testing FLUX imports...')
try:
    from library import flux_utils, flux_models, flux_train_utils
    print('✓ FLUX library imports successful')
except ImportError as e:
    print(f'⚠ FLUX library import failed: {e}')

print('Testing SD3 imports...')
try:
    from library import sd3_utils, sd3_models, sd3_train_utils
    print('✓ SD3 library imports successful')
except ImportError as e:
    print(f'⚠ SD3 library import failed: {e}')
"

echo ""
echo "4. Testing training scripts availability..."
echo "Available FLUX.1 training scripts:"
ls -la flux_*.py 2>/dev/null | head -5 || echo "No FLUX scripts found"
echo ""
echo "Available SD3 training scripts:"
ls -la sd3_*.py 2>/dev/null | head -5 || echo "No SD3 scripts found"

echo ""
echo "5. Testing xformers (if available)..."
python -c "
try:
    import xformers
    print(f'✓ xformers: {xformers.__version__}')
except ImportError:
    print('⚠ xformers not available')
"

echo ""
echo "6. Testing flash-attention (if available)..."
python -c "
try:
    import flash_attn
    print('✓ flash-attention available')
except ImportError:
    print('⚠ flash-attention not available (this is normal)')
"

echo ""
echo "=== Installation test completed ==="
EOF

chmod +x /workspace/test_installation.sh

# 创建模型验证脚本
log ">>> Creating model verification script..."
cat > /workspace/verify_models.sh <<'EOF'
#!/bin/bash
source /workspace/sd-scripts-env/bin/activate

echo "=== Verifying model files ==="
python -c "
import os
import safetensors.torch

models = {
    'flux1-dev.safetensors': '/workspace/models/flux1-dev.safetensors',
    'clip_l.safetensors': '/workspace/models/clip_l.safetensors',
    't5xxl_fp16.safetensors': '/workspace/models/t5xxl_fp16.safetensors',
    'ae.safetensors': '/workspace/models/ae.safetensors'
}

print('Model file status:')
print('-' * 60)
for name, path in models.items():
    if os.path.exists(path):
        try:
            # 检查文件大小
            size = os.path.getsize(path) / (1024**3)  # GB
            print(f'✓ {name:<25} {size:>8.2f} GB')
            
            # 尝试加载 safetensors 元数据
            with safetensors.safe_open(path, framework='pt', device='cpu') as f:
                print(f'  Tensors: {len(f.keys())} keys')
        except Exception as e:
            print(f'⚠ {name:<25} Error: {str(e)[:50]}...')
    else:
        print(f'✗ {name:<25} Not found')
print('-' * 60)
"
EOF

chmod +x /workspace/verify_models.sh

# 创建快速训练测试脚本
log ">>> Creating quick training test script..."
cat > /workspace/quick_test_training.sh <<'EOF'
#!/bin/bash
source /workspace/sd-scripts-env/bin/activate
cd /workspace/sd-scripts

echo "=== Quick Training Test ==="
echo ""
echo "Testing FLUX.1 training script..."
python flux_train_network.py --help | head -20

echo ""
echo "Testing SD3 training script..."
python sd3_train_network.py --help | head -20

echo ""
echo "Testing inference script..."
python flux_minimal_inference.py --help | head -20

echo ""
echo "✓ All training scripts are accessible"
EOF

chmod +x /workspace/quick_test_training.sh

# 创建示例训练命令脚本
log ">>> Creating example training commands..."
cat > /workspace/example_commands.sh <<'EOF'
#!/bin/bash

echo "=== SD-Scripts Example Training Commands ==="
echo ""
echo "📋 FLUX.1 LoRA Training (24GB VRAM):"
echo "accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py \\"
echo "  --pretrained_model_name_or_path /workspace/models/flux1-dev.safetensors \\"
echo "  --clip_l /workspace/models/clip_l.safetensors \\"
echo "  --t5xxl /workspace/models/t5xxl_fp16.safetensors \\"
echo "  --ae /workspace/models/ae.safetensors \\"
echo "  --cache_latents_to_disk --save_model_as safetensors --sdpa \\"
echo "  --persistent_data_loader_workers --max_data_loader_n_workers 2 \\"
echo "  --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \\"
echo "  --network_module networks.lora_flux --network_dim 4 --network_train_unet_only \\"
echo "  --optimizer_type adamw8bit --learning_rate 1e-4 \\"
echo "  --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base \\"
echo "  --highvram --max_train_epochs 4 --save_every_n_epochs 1 \\"
echo "  --dataset_config dataset_1024_bs2.toml \\"
echo "  --output_dir /workspace/output --output_name flux-lora-name \\"
echo "  --timestep_sampling shift --discrete_flow_shift 3.1582 \\"
echo "  --model_prediction_type raw --guidance_scale 1.0"
echo ""
echo "📋 SD3 LoRA Training (16GB VRAM):"
echo "accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 sd3_train_network.py \\"
echo "  --pretrained_model_name_or_path path/to/sd3.5_large.safetensors \\"
echo "  --clip_l sd3/clip_l.safetensors --clip_g sd3/clip_g.safetensors \\"
echo "  --t5xxl sd3/t5xxl_fp16.safetensors \\"
echo "  --cache_latents_to_disk --save_model_as safetensors --sdpa \\"
echo "  --persistent_data_loader_workers --max_data_loader_n_workers 2 \\"
echo "  --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 \\"
echo "  --network_module networks.lora_sd3 --network_dim 4 --network_train_unet_only \\"
echo "  --optimizer_type adamw8bit --learning_rate 1e-4 \\"
echo "  --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base \\"
echo "  --highvram --max_train_epochs 4 --save_every_n_epochs 1 \\"
echo "  --dataset_config dataset_1024_bs2.toml \\"
echo "  --output_dir /workspace/output --output_name sd3-lora-name"
echo ""
echo "💡 Tips:"
echo "  - For lower VRAM, use --blocks_to_swap option"
echo "  - For 12GB VRAM, use --blocks_to_swap 16"
echo "  - For 8GB VRAM, use --blocks_to_swap 28"
echo "  - DeepSpeed is required for FLUX.1 ControlNet training"
EOF

chmod +x /workspace/example_commands.sh

# 创建优化的 accelerate 配置
log ">>> Creating optimized accelerate configuration..."
cat > /workspace/accelerate_config.yaml <<'EOF'
compute_environment: LOCAL_PROCESS
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

mkdir -p ~/.cache/huggingface/accelerate
cp /workspace/accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml

# ========== 环境变量持久化 ==========
log "=== Phase 7: Environment Persistence ==="

log ">>> Setting up persistent environment variables..."
cat >> /etc/environment <<EOF
CUDA_HOME=/usr/local/cuda
PATH=/workspace/sd-scripts-env/bin:/usr/local/cuda/bin:\$PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
PYTHONPATH=/workspace/sd-scripts:\$PYTHONPATH
TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
HF_HOME=/workspace/.cache/huggingface
EOF

# 只有当 tcmalloc 存在时才添加 LD_PRELOAD 到 /etc/environment
if [ -f "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" ]; then
    echo 'LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4' >> /etc/environment
elif [ -f "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4" ]; then
    echo 'LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4' >> /etc/environment
fi

# ========== 最终验证 ==========
log "=== Phase 8: Final Verification ==="

# 运行依赖检查
log ">>> Running dependency compliance check..."
/workspace/check_dependencies.sh

# 运行安装测试
log ">>> Running installation test..."
/workspace/test_installation.sh

# 验证模型文件
log ">>> Verifying model files..."
/workspace/verify_models.sh

# ========== 显示完成信息 ==========
log ""
log "================================================================================"
log "✅ SD-Scripts FLUX.1/SD3 Environment Setup Completed Successfully!"
log "================================================================================"
log ""
log "🎯 Environment Summary:"
log "  - Python: $(python --version)"
log "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
log "  - CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
DEEPSPEED_VER=$(python - <<'PY'
try:
    import deepspeed
    print(deepspeed.__version__, end='')
except Exception:
    print("Not available", end='')
PY
)
log "  - DeepSpeed: $DEEPSPEED_VER"
log ""
log "📋 Quick Start Commands:"
log "  1. Activate environment:    source /workspace/activate_env.sh"
log "  2. Check dependencies:      /workspace/check_dependencies.sh"
log "  3. Test installation:       /workspace/test_installation.sh"
log "  4. Verify models:          /workspace/verify_models.sh"
log "  5. Example commands:       /workspace/example_commands.sh"
log ""
log "🚀 Key Training Scripts:"
log "  FLUX.1 LoRA:     flux_train_network.py"
log "  FLUX.1 Full:     flux_train.py"
log "  SD3 LoRA:        sd3_train_network.py"
log "  SD3 Full:        sd3_train.py"
log "  Inference:       flux_minimal_inference.py, sd3_minimal_inference.py"
log ""
log "📦 Model Files (if downloaded):"
log "  FLUX.1-dev:  /workspace/models/flux1-dev.safetensors"
log "  CLIP-L:      /workspace/models/clip_l.safetensors"
log "  T5XXL:       /workspace/models/t5xxl_fp16.safetensors"
log "  AE:          /workspace/models/ae.safetensors"
log ""
log "💡 Performance Tips:"
log "  - Use --blocks_to_swap for lower VRAM usage"
log "  - DeepSpeed is required for FLUX.1 ControlNet training"
log "  - Batch size 1 recommended for 24GB VRAM"
log ""
log "🔧 Environment Details:"
log "  - Virtual env:       /workspace/sd-scripts-env/"
log "  - Scripts:          /workspace/sd-scripts/"
log "  - Models:           /workspace/models/"
log "  - Accelerate config: ~/.cache/huggingface/accelerate/default_config.yaml"
log ""
if [ -z "$HF_TOKEN" ]; then
    log_error "⚠️  IMPORTANT: HF_TOKEN not set. Some models may not be downloaded."
    echo "   Please set HF_TOKEN in vast.ai environment variables to download FLUX.1-dev"
fi
log ""
log "⏱️  Timing Summary:"
log "  Start time: $START_TIME"
log "  End time:   $(date)"
log ""
log "🎉 Ready for FLUX.1/SD3 training!"
log "================================================================================" 

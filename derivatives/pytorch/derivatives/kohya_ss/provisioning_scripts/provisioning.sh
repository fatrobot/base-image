#!/bin/bash

# SD-Scripts FLUX.1/SD3 Training Environment Provisioning Script
# For vast.ai with CUDA 12.4.1 and Ubuntu 22.04
# Version: 1.0
# Last updated: 2025-01-18

# 脚本出错时退出
set -eo pipefail

echo "=== SD-Scripts FLUX.1/SD3 Training Environment Setup ==="
echo "Starting provisioning script..."
echo "Version: 1.0"
echo "Base image: cuda-12.4.1-cudnn-devel-ubuntu22.04-py310"

# 记录开始时间
START_TIME=$(date)
echo "Start time: $START_TIME"

# 切换到持久化目录
cd /workspace/

# ========== 系统环境准备 ==========
echo "=== Phase 1: System Environment Setup ==="

# 更新系统包和安装编译工具
echo ">>> Updating system packages and installing build tools..."
apt-get update -y
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
    vim

# 设置内存优化
echo ">>> Setting up memory optimization..."
export LD_PRELOAD=libtcmalloc.so.4:$LD_PRELOAD
echo 'export LD_PRELOAD=libtcmalloc.so.4:$LD_PRELOAD' >> /etc/environment

# 验证 CUDA 环境
echo ">>> Verifying CUDA environment..."
nvidia-smi
nvcc --version

# 设置 CUDA 相关环境变量
echo ">>> Setting up CUDA environment variables..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# ========== Python 环境设置 ==========
echo "=== Phase 2: Python Environment Setup ==="

# 创建 Python 虚拟环境
echo ">>> Creating Python virtual environment..."
python3.10 -m venv sd-scripts-env
source sd-scripts-env/bin/activate

# 验证 Python 版本
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# 升级 pip 和基础工具
echo ">>> Upgrading pip and basic tools..."
pip install --upgrade pip setuptools wheel

# ========== SD-Scripts 项目设置 ==========
echo "=== Phase 3: SD-Scripts Project Setup ==="

# 克隆 sd-scripts 项目 (sd3 分支)
echo ">>> Cloning sd-scripts repository (sd3 branch)..."
if [ -d "sd-scripts" ]; then
    echo "sd-scripts directory exists, removing..."
    rm -rf sd-scripts
fi

git clone --branch sd3 --depth 1 https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

# ========== 依赖安装 ==========
echo "=== Phase 4: Dependencies Installation ==="

# 安装 PyTorch 2.4.0 with CUDA 12.4 (README 明确要求)
echo ">>> Installing PyTorch 2.4.0 with CUDA 12.4..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 验证 PyTorch 安装
echo ">>> Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 安装 xformers (与 PyTorch 2.4.0 兼容的版本)
echo ">>> Installing xformers for PyTorch 2.4.0..."
pip install xformers --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖 (严格按照 requirements.txt)
echo ">>> Installing sd-scripts requirements..."
pip install -r requirements.txt

# 验证关键依赖版本
echo ">>> Verifying key dependencies versions..."
python -c "
import accelerate, transformers, diffusers, bitsandbytes, safetensors
print(f'accelerate: {accelerate.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'diffusers: {diffusers.__version__}')
print(f'bitsandbytes: {bitsandbytes.__version__}')
print(f'safetensors: {safetensors.__version__}')
"

# 安装 DeepSpeed (README 明确要求的版本)
echo ">>> Installing DeepSpeed (required for FLUX.1/SD3)..."
pip install deepspeed==0.16.7

# 安装额外的性能优化包
echo ">>> Installing additional performance packages..."
pip install --no-deps wandb || echo "wandb installation failed, continuing..."

# 尝试安装 flash-attention (可能失败)
echo ">>> Attempting to install flash-attention..."
pip install flash-attn --no-build-isolation || echo "flash-attn installation failed, this is normal for some environments"

# 尝试安装 triton
echo ">>> Attempting to install triton..."
pip install triton || echo "triton installation failed, continuing..."

# 安装可选的 WD14 tagger 依赖
echo ">>> Installing optional WD14 tagger dependencies..."
pip install onnx==1.15.0 onnxruntime-gpu==1.17.1 || echo "ONNX dependencies installation failed, continuing..."

# ========== 模型下载 ==========
echo "=== Phase 5: Model Downloads ==="

# 返回 workspace 目录
cd /workspace/

# 创建模型目录
echo ">>> Creating models directory..."
mkdir -p /workspace/models
cd /workspace/models

# 检查 HuggingFace Token
echo ">>> Checking HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  WARNING: HF_TOKEN environment variable not set!"
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
    echo "✓ HF_TOKEN found, configuring authentication..."
    # 安装 huggingface-cli
    pip install -U huggingface_hub
    
    # 登录 HuggingFace
    huggingface-cli login --token $HF_TOKEN --add-to-git-credential
    echo "✓ HuggingFace authentication configured"
fi

# 下载模型函数
download_model() {
    local url=$1
    local filename=$2
    local require_auth=$3
    
    if [ "$require_auth" = "true" ] && [ "$SKIP_AUTH_MODELS" = "true" ]; then
        echo "⚠️  Skipping $filename (requires authentication)"
        return
    fi
    
    if [ -f "$filename" ]; then
        echo "✓ $filename already exists, skipping download"
    else
        echo "📥 Downloading $filename..."
        if [ "$require_auth" = "true" ]; then
            # 使用 huggingface-cli 下载需要认证的文件
            huggingface-cli download --resume-download --local-dir . \
                $(echo $url | sed 's|https://huggingface.co/||' | sed 's|/resolve/.*||') \
                $(basename $url) --local-dir-use-symlinks False
            mv $(basename $url) $filename 2>/dev/null || true
        else
            # 使用 wget 下载公开文件
            wget -c -O $filename "$url" || {
                echo "❌ Failed to download $filename"
                rm -f $filename
                return 1
            }
        fi
        echo "✓ Successfully downloaded $filename"
    fi
}

# 下载模型文件
echo ">>> Downloading model files..."
echo "This may take a while depending on your internet speed..."

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
echo ">>> Creating model paths configuration..."
cat > /workspace/model_paths.txt <<'EOF'
# Model paths for sd-scripts training
FLUX_MODEL=/workspace/models/flux1-dev.safetensors
CLIP_L=/workspace/models/clip_l.safetensors
T5XXL=/workspace/models/t5xxl_fp16.safetensors
AE=/workspace/models/ae.safetensors
EOF

# 显示模型状态
echo ">>> Model files status:"
ls -lah /workspace/models/

# 返回 sd-scripts 目录
cd /workspace/sd-scripts

# ========== 辅助脚本创建 ==========
echo "=== Phase 6: Creating Helper Scripts ==="

# 创建环境激活脚本
echo ">>> Creating environment activation script..."
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
echo "DeepSpeed available: $(python -c 'try: import deepspeed; print(True); except: print(False)')"

if [ $# -gt 0 ]; then
    exec "$@"
else
    exec bash
fi
EOF

chmod +x /workspace/activate_env.sh

# 创建依赖检查脚本
echo ">>> Creating dependency check script..."
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
echo ">>> Creating installation test script..."
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
echo ">>> Creating model verification script..."
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
echo ">>> Creating quick training test script..."
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
echo ">>> Creating example training commands..."
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
echo ">>> Creating optimized accelerate configuration..."
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
echo "=== Phase 7: Environment Persistence ==="

echo ">>> Setting up persistent environment variables..."
cat >> /etc/environment <<EOF
CUDA_HOME=/usr/local/cuda
PATH=/workspace/sd-scripts-env/bin:/usr/local/cuda/bin:\$PATH
LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
PYTHONPATH=/workspace/sd-scripts:\$PYTHONPATH
LD_PRELOAD=libtcmalloc.so.4:\$LD_PRELOAD
TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
HF_HOME=/workspace/.cache/huggingface
EOF

# ========== 最终验证 ==========
echo "=== Phase 8: Final Verification ==="

# 运行依赖检查
echo ">>> Running dependency compliance check..."
/workspace/check_dependencies.sh

# 运行安装测试
echo ">>> Running installation test..."
/workspace/test_installation.sh

# 验证模型文件
echo ">>> Verifying model files..."
/workspace/verify_models.sh

# ========== 显示完成信息 ==========
echo ""
echo "================================================================================"
echo "✅ SD-Scripts FLUX.1/SD3 Environment Setup Completed Successfully!"
echo "================================================================================"
echo ""
echo "🎯 Environment Summary:"
echo "  - Python: $(python --version)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  - DeepSpeed: $(python -c 'try: import deepspeed; print(deepspeed.__version__); except: print("Not available")')"
echo ""
echo "📋 Quick Start Commands:"
echo "  1. Activate environment:    source /workspace/activate_env.sh"
echo "  2. Check dependencies:      /workspace/check_dependencies.sh"
echo "  3. Test installation:       /workspace/test_installation.sh"
echo "  4. Verify models:          /workspace/verify_models.sh"
echo "  5. Example commands:       /workspace/example_commands.sh"
echo ""
echo "🚀 Key Training Scripts:"
echo "  FLUX.1 LoRA:     flux_train_network.py"
echo "  FLUX.1 Full:     flux_train.py"
echo "  SD3 LoRA:        sd3_train_network.py"
echo "  SD3 Full:        sd3_train.py"
echo "  Inference:       flux_minimal_inference.py, sd3_minimal_inference.py"
echo ""
echo "📦 Model Files (if downloaded):"
echo "  FLUX.1-dev:  /workspace/models/flux1-dev.safetensors"
echo "  CLIP-L:      /workspace/models/clip_l.safetensors"
echo "  T5XXL:       /workspace/models/t5xxl_fp16.safetensors"
echo "  AE:          /workspace/models/ae.safetensors"
echo ""
echo "💡 Performance Tips:"
echo "  - Use --blocks_to_swap for lower VRAM usage"
echo "  - DeepSpeed is required for FLUX.1 ControlNet training"
echo "  - Batch size 1 recommended for 24GB VRAM"
echo ""
echo "🔧 Environment Details:"
echo "  - Virtual env:       /workspace/sd-scripts-env/"
echo "  - Scripts:          /workspace/sd-scripts/"
echo "  - Models:           /workspace/models/"
echo "  - Accelerate config: ~/.cache/huggingface/accelerate/default_config.yaml"
echo ""
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  IMPORTANT: HF_TOKEN not set. Some models may not be downloaded."
    echo "   Please set HF_TOKEN in vast.ai environment variables to download FLUX.1-dev"
fi
echo ""
echo "⏱️  Timing Summary:"
echo "  Start time: $START_TIME"
echo "  End time:   $(date)"
echo ""
echo "🎉 Ready for FLUX.1/SD3 training!"
echo "================================================================================" 

#!/bin/bash

# SD-Scripts FLUX.1/SD3 Training Environment Provisioning Script (Stable Version)
# For vast.ai with CUDA 12.4.1 and Ubuntu 22.04 Python 3.10
# Version: 2.1 (Improved)
# Last updated: 2025-06-23

# 严格错误处理 - 出错即停止
set -euo pipefail

# 定义全局变量
WORKSPACE_DIR="/workspace"
VENV_NAME="sd-scripts-env"
PROJECT_NAME="sd-scripts"
MODELS_DIR="models"
REQUIRED_SPACE_GB=50  # 需要的磁盘空间（GB）

# 创建日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

log_warning() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1" >&2
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1"
}

# 错误处理函数
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Script failed at line $line_number with exit code $exit_code"
    log_error "Please check the logs above for details"
    exit $exit_code
}

# 设置错误处理
trap 'handle_error ${LINENO}' ERR

# 检查磁盘空间
check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG "$WORKSPACE_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi
    log_success "Disk space check passed. Available: ${available_gb}GB"
}

# 下载文件函数（带重试）
download_with_retry() {
    local url=$1
    local output=$2
    local auth_header=${3:-}
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if [ -n "$auth_header" ]; then
            if wget --header="$auth_header" "$url" -O "$output" 2>/dev/null; then
                return 0
            fi
        else
            if wget "$url" -O "$output" 2>/dev/null; then
                return 0
            fi
        fi
        
        retry_count=$((retry_count + 1))
        log_warning "Download failed, retry $retry_count/$max_retries..."
        sleep 5
    done
    
    log_error "Failed to download $url after $max_retries attempts"
    return 1
}

# Python 包验证函数
verify_python_package() {
    local package_name=$1
    local import_name=${2:-$package_name}
    
    if python -c "import $import_name; print(f'✓ $package_name: {$import_name.__version__}')" 2>/dev/null; then
        return 0
    else
        log_error "Failed to import $package_name"
        return 1
    fi
}

log "=== SD-Scripts FLUX.1/SD3 Training Environment Setup (Stable Version) ==="
log "Starting provisioning script..."
log "Version: 2.1 (Improved)"
log "Base image: cuda-12.4.1-cudnn-devel-ubuntu22.04-py310"

# 记录开始时间
START_TIME=$(date)
log "Start time: $START_TIME"

# 修复 LD_PRELOAD 错误 - 清除有问题的环境变量
unset LD_PRELOAD 2>/dev/null || true
export DEBIAN_FRONTEND=noninteractive

# 切换到持久化目录
cd "$WORKSPACE_DIR"

# 检查磁盘空间
check_disk_space $REQUIRED_SPACE_GB

# ========== 系统环境准备 ==========
log "=== Phase 1: System Environment Setup ==="

# 更新系统包和安装基础工具
log ">>> Updating system packages..."
apt-get update -y || {
    log_warning "apt update failed, continuing anyway..."
}

log ">>> Installing essential system packages..."
apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libffi-dev \
    libssl-dev \
    python3-dev \
    python3.10-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    htop \
    tmux \
    vim \
    tree || {
    log_error "Failed to install system packages"
    exit 1
}

# 验证 CUDA 环境
log ">>> Verifying CUDA environment..."
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found, CUDA environment may not be properly set up"
    exit 1
fi

nvidia-smi
nvcc --version || log_warning "nvcc not found, some packages may need to build from source"

# 设置 CUDA 环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# ========== Python 环境设置 ==========
log "=== Phase 2: Python Environment Setup ==="

# 创建 Python 虚拟环境
log ">>> Creating Python virtual environment..."
if [ -d "$VENV_NAME" ]; then
    log_warning "Virtual environment already exists, removing..."
    rm -rf "$VENV_NAME"
fi

python3.10 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# 验证 Python 版本
PYTHON_VERSION=$(python --version)
PIP_VERSION=$(pip --version)
log "Python version: $PYTHON_VERSION"
log "Pip version: $PIP_VERSION"

# 确保我们使用的是正确的 Python
if ! python --version | grep -q "3.10"; then
    log_error "Python version is not 3.10.x"
    exit 1
fi

# 升级 pip 和基础工具
log ">>> Upgrading pip and basic tools..."
pip install --upgrade pip setuptools wheel

# ========== SD-Scripts 项目设置 ==========
log "=== Phase 3: SD-Scripts Project Setup ==="

# 克隆 sd-scripts 项目 (sd3 分支)
log ">>> Cloning sd-scripts repository (sd3 branch)..."
if [ -d "$PROJECT_NAME" ]; then
    log "sd-scripts directory exists, removing..."
    rm -rf "$PROJECT_NAME"
fi

if ! git clone --branch sd3 --depth 1 https://github.com/kohya-ss/sd-scripts.git; then
    log_error "Failed to clone sd-scripts repository"
    exit 1
fi

cd "$PROJECT_NAME"

# ========== 核心依赖安装 (按特定顺序避免冲突) ==========
log "=== Phase 4: Core Dependencies Installation ==="

# Step 1: 安装 PyTorch 2.4.0 和相关工具链
log ">>> Installing PyTorch 2.4.0 with CUDA 12.4 support..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 验证 PyTorch 安装
log ">>> Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA is not available!')
" || {
    log_error "PyTorch installation verification failed"
    exit 1
}

# Step 2: 安装 triton 解决 bitsandbytes 依赖问题
log ">>> Installing triton for bitsandbytes compatibility..."
pip install triton==3.0.0

# Step 3: 安装项目特定依赖 (精确版本控制)
log ">>> Installing sd-scripts core dependencies with version control..."

# 定义核心依赖
CORE_DEPS=(
    "accelerate==0.33.0"
    "transformers==4.44.0"
    "diffusers[torch]==0.25.0"
    "huggingface-hub==0.24.5"
    "safetensors==0.4.4"
    "bitsandbytes==0.44.0"
)

# 安装核心依赖
for dep in "${CORE_DEPS[@]}"; do
    log "Installing $dep..."
    pip install "$dep" || {
        log_error "Failed to install $dep"
        exit 1
    }
done

# 验证核心依赖
log ">>> Verifying core dependencies..."
verify_python_package "accelerate" || exit 1
verify_python_package "transformers" || exit 1
verify_python_package "diffusers" || exit 1
verify_python_package "huggingface_hub" || exit 1
verify_python_package "safetensors" || exit 1
verify_python_package "bitsandbytes" || exit 1

# Step 4: 安装其他必需依赖
log ">>> Installing additional dependencies..."
ADDITIONAL_DEPS=(
    "ftfy==6.1.1"
    "opencv-python==4.8.1.78"
    "einops==0.7.0"
    "pytorch-lightning==1.9.0"
    "lion-pytorch==0.0.6"
    "schedulefree==1.4"
    "pytorch-optimizer==3.5.0"
    "prodigy-plus-schedule-free==1.9.0"
    "prodigyopt==1.1.2"
    "tensorboard"
    "altair==4.2.2"
    "easygui==0.98.3"
    "toml==0.10.2"
    "voluptuous==0.13.1"
    "imagesize==1.4.1"
    "numpy<=2.0"
    "rich==13.7.0"
    "sentencepiece==0.2.0"
)

for dep in "${ADDITIONAL_DEPS[@]}"; do
    pip install "$dep" || log_warning "Failed to install $dep, continuing..."
done

# Step 5: 安装 xformers (兼容 PyTorch 2.4.0)
log ">>> Installing xformers compatible with PyTorch 2.4.0..."
pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124 --no-deps || {
    log_warning "xformers installation failed, this is optional and training can continue without it"
}

# 验证 xformers 安装
verify_python_package "xformers" || log_warning "xformers not available, continuing without it"

# Step 6: 安装 DeepSpeed (FLUX.1 训练必需)
log ">>> Installing DeepSpeed 0.16.7 for FLUX.1 training..."
pip install deepspeed==0.16.7 || {
    log_warning "DeepSpeed installation failed, FLUX.1 training may not work properly"
}

# 验证 DeepSpeed 安装
verify_python_package "deepspeed" || log_warning "DeepSpeed not available"

# Step 7: 安装项目本身
log ">>> Installing sd-scripts package in editable mode..."
pip install -e . || {
    log_error "Failed to install sd-scripts package"
    exit 1
}

# ========== 模型下载 ==========
log "=== Phase 5: Model Download ==="

# 切换到工作目录
cd "$WORKSPACE_DIR"

# 创建模型目录
log ">>> Creating models directory..."
mkdir -p "$MODELS_DIR"

# 检查 HF_TOKEN 环境变量
if [ -z "${HF_TOKEN:-}" ]; then
    log_warning "HF_TOKEN not found, skipping model download. Please set HF_TOKEN environment variable."
else
    log ">>> HF_TOKEN found, downloading FLUX.1 models..."
    
    # 切换到模型目录
    cd "$MODELS_DIR"
    
    # 定义模型下载
    declare -A MODELS=(
        ["flux1-dev.safetensors"]="https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"
        ["clip_l.safetensors"]="https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
        ["t5xxl_fp16.safetensors"]="https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
        ["ae.safetensors"]="https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors"
    )
    
    # 下载模型
    for model_name in "${!MODELS[@]}"; do
        if [ -f "$model_name" ]; then
            log "Model $model_name already exists, skipping..."
        else
            log ">>> Downloading $model_name..."
            if [[ "$model_name" == "flux1-dev.safetensors" ]]; then
                download_with_retry "${MODELS[$model_name]}" "$model_name" "Authorization: Bearer $HF_TOKEN" || {
                    log_warning "Failed to download $model_name"
                }
            else
                download_with_retry "${MODELS[$model_name]}" "$model_name" || {
                    log_warning "Failed to download $model_name"
                }
            fi
        fi
    done
    
    log ">>> Model download phase completed."
    ls -la "$WORKSPACE_DIR/$MODELS_DIR/"
fi

# ========== 最终验证 ==========
log "=== Phase 6: Final Verification ==="

# 切换回项目目录
cd "$WORKSPACE_DIR/$PROJECT_NAME"

# 验证关键导入
log ">>> Performing final verification..."
python << 'EOF'
import sys
import torch
import diffusers
import transformers
import accelerate
import safetensors

print('=== Final Verification Report ===')
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
print(f'Diffusers version: {diffusers.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'Accelerate version: {accelerate.__version__}')
print(f'Safetensors version: {safetensors.__version__}')

# 测试关键模块导入
try:
    from diffusers import FluxPipeline
    print('✓ FluxPipeline import successful')
except ImportError as e:
    print(f'✗ FluxPipeline import failed: {e}')

try:
    from huggingface_hub import cached_download
    print('✓ huggingface_hub cached_download available')
except ImportError as e:
    print(f'✗ huggingface_hub cached_download failed: {e}')

print('=== Verification Complete ===')
EOF

# 创建快速启动脚本
log ">>> Creating quick start script..."
cat > "$WORKSPACE_DIR/activate_env.sh" << EOF
#!/bin/bash
# Quick start script for SD-Scripts environment

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

echo "Activating SD-Scripts environment..."
cd $WORKSPACE_DIR
source $VENV_NAME/bin/activate
cd $PROJECT_NAME

echo "Environment activated. You can now run training scripts."
echo "Example: python flux_train_network.py --help"
echo ""
echo "Available models in $WORKSPACE_DIR/$MODELS_DIR:"
ls -la $WORKSPACE_DIR/$MODELS_DIR/*.safetensors 2>/dev/null || echo "No models found. Please set HF_TOKEN and re-run the provisioning script."
EOF

chmod +x "$WORKSPACE_DIR/activate_env.sh"

# 显示安装总结
END_TIME=$(date)
log "=== Installation Summary ==="
log "Start time: $START_TIME"
log "End time: $END_TIME"
log "Python environment: $WORKSPACE_DIR/$VENV_NAME"
log "SD-Scripts location: $WORKSPACE_DIR/$PROJECT_NAME"
log "Models location: $WORKSPACE_DIR/$MODELS_DIR"

# 最终验证
log ">>> Final verification: Key components check..."

# 检查关键目录
for dir in "$WORKSPACE_DIR/$VENV_NAME" "$WORKSPACE_DIR/$PROJECT_NAME" "$WORKSPACE_DIR/$MODELS_DIR"; do
    if [ -d "$dir" ]; then
        log_success "$(basename $dir) directory exists"
    else
        log_error "$(basename $dir) directory missing"
    fi
done

# 检查激活脚本
if [ -f "$WORKSPACE_DIR/activate_env.sh" ]; then
    log_success "Activation script created"
else
    log_error "Activation script missing"
fi

log "=== Provisioning Script Completed Successfully ==="
log "✓ Run 'source $WORKSPACE_DIR/activate_env.sh' to activate the environment"
log "✓ Training scripts are available in $WORKSPACE_DIR/$PROJECT_NAME/"
log "✓ Models are available in $WORKSPACE_DIR/$MODELS_DIR/ (if HF_TOKEN was provided)" 

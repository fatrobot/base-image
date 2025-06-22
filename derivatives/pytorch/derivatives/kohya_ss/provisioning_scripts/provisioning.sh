#!/bin/bash

# SD-Scripts FLUX.1/SD3 Training Environment Provisioning Script (Stable Version)
# For vast.ai with CUDA 12.4.1 and Ubuntu 22.04 Python 3.10
# Version: 2.2 (Security & Robustness Enhanced)
# Last updated: 2025-01-23

# 严格错误处理 - 出错即停止
set -euo pipefail

# 定义全局变量
WORKSPACE_DIR="/workspace"
VENV_NAME="sd-scripts-env"
PROJECT_NAME="sd-scripts"
MODELS_DIR="models"
REQUIRED_SPACE_GB=50
LOG_FILE="$WORKSPACE_DIR/provisioning_$(date +%Y%m%d_%H%M%S).log"

# 创建日志函数
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$LOG_FILE"
}

log_error() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo "$message" >&2 | tee -a "$LOG_FILE"
}

log_warning() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1"
    echo "$message" >&2 | tee -a "$LOG_FILE"
}

log_success() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1"
    echo "$message" | tee -a "$LOG_FILE"
}

# 错误处理函数
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Script failed at line $line_number with exit code $exit_code"
    log_error "Check log file: $LOG_FILE"
    cleanup_on_error
    exit $exit_code
}

# 清理函数
cleanup_on_error() {
    log "Cleaning up temporary files..."
    find "${WORKSPACE_DIR}/${MODELS_DIR}" -name "*.tmp" -o -name "*.part" -delete 2>/dev/null || true
    rm -f ~/.wgetrc_tmp 2>/dev/null || true
}

# 设置错误处理
trap 'handle_error ${LINENO}' ERR
trap cleanup_on_error EXIT

# 检查磁盘空间（更健壮的实现）
check_disk_space() {
    local required_gb=$1
    local workspace_dir=$2
    
    # 使用更可靠的方式获取可用空间
    local available_kb=$(df -k "$workspace_dir" | tail -1 | awk '{print $4}')
    local available_gb=$((available_kb / 1024 / 1024))
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi
    
    # 添加10%缓冲
    local buffer_gb=$((required_gb * 11 / 10))
    if [ "$available_gb" -lt "$buffer_gb" ]; then
        log_warning "Disk space is tight. Available: ${available_gb}GB, Recommended: ${buffer_gb}GB"
    fi
    
    log_success "Disk space check passed. Available: ${available_gb}GB"
}

# 安全的下载函数
download_with_retry() {
    local url=$1
    local output=$2
    local auth_token=${3:-}
    local max_retries=3
    local retry_count=0
    local temp_file="${output}.tmp"
    
    rm -f "$temp_file"
    
    while [ $retry_count -lt $max_retries ]; do
        log "Downloading $(basename "$output") (attempt $((retry_count + 1))/$max_retries)..."
        
        # 构建wget命令，显示进度
        local wget_cmd="wget --progress=bar:force --show-progress"
        
        # 如果有部分下载，尝试续传
        if [ -f "$output.part" ]; then
            wget_cmd="$wget_cmd -c"
            temp_file="$output.part"
        fi
        
        # 使用临时配置文件传递认证信息（更安全）
        if [ -n "$auth_token" ]; then
            echo "header = Authorization: Bearer $auth_token" > ~/.wgetrc_tmp
            wget_cmd="$wget_cmd --config ~/.wgetrc_tmp"
        fi
        
        # 执行下载
        if $wget_cmd "$url" -O "$temp_file" 2>&1 | tee -a "$LOG_FILE"; then
            mv "$temp_file" "$output"
            rm -f ~/.wgetrc_tmp 2>/dev/null || true
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            log_warning "Download failed, waiting 10 seconds before retry..."
            sleep 10
        fi
    done
    
    rm -f ~/.wgetrc_tmp "$temp_file" 2>/dev/null || true
    log_error "Failed to download $url after $max_retries attempts"
    return 1
}

# Python包验证函数
verify_python_package() {
    local package_name=$1
    local import_name=${2:-$package_name}
    local required=${3:-true}
    
    if python -c "import $import_name; print(f'✓ $package_name: {$import_name.__version__}')" 2>>"$LOG_FILE"; then
        return 0
    else
        if [ "$required" = "true" ]; then
            log_error "Failed to import required package: $package_name"
            return 1
        else
            log_warning "Failed to import optional package: $package_name"
            return 0
        fi
    fi
}

# 验证虚拟环境激活
verify_venv_activated() {
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        log_error "Virtual environment is not activated"
        return 1
    fi
    
    if [ "$(which python)" != "$VIRTUAL_ENV/bin/python" ]; then
        log_error "Python is not from virtual environment"
        return 1
    fi
    
    return 0
}

# 开始执行
log "=== SD-Scripts FLUX.1/SD3 Training Environment Setup ==="
log "Version: 2.2 (Security & Robustness Enhanced)"
log "Log file: $LOG_FILE"

START_TIME=$(date)
START_TIMESTAMP=$(date +%s)
log "Start time: $START_TIME"

# 清理环境变量
unset LD_PRELOAD 2>/dev/null || true
export DEBIAN_FRONTEND=noninteractive

# 切换到工作目录
cd "$WORKSPACE_DIR"

# 检查磁盘空间
check_disk_space $REQUIRED_SPACE_GB "$WORKSPACE_DIR"

# ========== 系统环境准备 ==========
log "=== Phase 1: System Environment Setup ==="

log ">>> Updating system packages..."
apt-get update -y 2>&1 | tee -a "$LOG_FILE" || log_warning "apt update had issues"

log ">>> Installing essential system packages..."
SYSTEM_PACKAGES=(
    git wget curl unzip
    build-essential cmake ninja-build
    pkg-config libffi-dev libssl-dev
    python3-dev python3.10-venv python3.10-distutils
    libgl1-mesa-glx libglib2.0-0 libsm6
    libxext6 libxrender-dev libgomp1
    htop tmux vim tree
)

apt-get install -y "${SYSTEM_PACKAGES[@]}" 2>&1 | tee -a "$LOG_FILE"

# 验证CUDA环境
log ">>> Verifying CUDA environment..."
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found"
    exit 1
fi

nvidia-smi | tee -a "$LOG_FILE"
nvcc --version 2>&1 | tee -a "$LOG_FILE" || log_warning "nvcc not found"

# 设置CUDA环境变量
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# ========== Python环境设置 ==========
log "=== Phase 2: Python Environment Setup ==="

log ">>> Creating Python virtual environment..."
[ -d "$VENV_NAME" ] && rm -rf "$VENV_NAME"

python3.10 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# 验证虚拟环境
verify_venv_activated

# 验证Python版本（精确匹配）
PYTHON_VERSION=$(python --version 2>&1)
log "Python version: $PYTHON_VERSION"

if ! echo "$PYTHON_VERSION" | grep -E "Python 3\.10\.[0-9]+" > /dev/null; then
    log_error "Python version mismatch: $PYTHON_VERSION"
    exit 1
fi

# 升级pip
log ">>> Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel 2>&1 | tee -a "$LOG_FILE"

# ========== SD-Scripts项目设置 ==========
log "=== Phase 3: SD-Scripts Project Setup ==="

log ">>> Cloning sd-scripts repository..."
[ -d "$PROJECT_NAME" ] && rm -rf "$PROJECT_NAME"

git clone --branch sd3 --depth 1 https://github.com/kohya-ss/sd-scripts.git 2>&1 | tee -a "$LOG_FILE"
cd "$PROJECT_NAME"

# ========== 核心依赖安装 ==========
log "=== Phase 4: Core Dependencies Installation ==="

# PyTorch安装
log ">>> Installing PyTorch 2.4.0..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tee -a "$LOG_FILE"

# 验证PyTorch
log ">>> Verifying PyTorch..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)')
" 2>&1 | tee -a "$LOG_FILE"

# 安装triton
pip install triton==3.0.0 2>&1 | tee -a "$LOG_FILE"

# 按顺序安装核心依赖
CORE_DEPS_ORDER=(
    "accelerate==0.33.0"
    "transformers==4.44.0"
    "diffusers[torch]==0.25.0"
    "huggingface-hub==0.24.5"
    "safetensors==0.4.4"
    "bitsandbytes==0.44.0"
)

for dep in "${CORE_DEPS_ORDER[@]}"; do
    log "Installing $dep..."
    pip install "$dep" 2>&1 | tee -a "$LOG_FILE"
done

# 验证核心依赖
for pkg in accelerate transformers diffusers huggingface_hub safetensors bitsandbytes; do
    verify_python_package "$pkg" "$pkg" "true" || exit 1
done

# 安装其他依赖
log ">>> Installing additional dependencies..."
pip install -r requirements.txt 2>&1 | tee -a "$LOG_FILE" || log_warning "Some optional deps failed"

# xformers安装
pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124 \
    --no-deps 2>&1 | tee -a "$LOG_FILE" || log_warning "xformers failed"

# DeepSpeed安装
pip install deepspeed==0.16.7 2>&1 | tee -a "$LOG_FILE" || log_warning "DeepSpeed failed"

# 安装项目
pip install -e . 2>&1 | tee -a "$LOG_FILE"

# ========== 模型下载 ==========
log "=== Phase 5: Model Download ==="

cd "$WORKSPACE_DIR"
mkdir -p "$MODELS_DIR"

if [ -z "${HF_TOKEN:-}" ]; then
    log_warning "HF_TOKEN not found, skipping model download"
else
    log ">>> Downloading models..."
    cd "$MODELS_DIR"
    
    # 模型列表（保持顺序）
    MODELS=(
        "flux1-dev.safetensors|https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors|auth"
        "clip_l.safetensors|https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors|"
        "t5xxl_fp16.safetensors|https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors|"
        "ae.safetensors|https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors|"
    )
    
    # 串行下载（避免并发问题）
    for model_info in "${MODELS[@]}"; do
        IFS='|' read -r name url auth_flag <<< "$model_info"
        
        if [ -f "$name" ]; then
            log "Model $name exists, skipping..."
            continue
        fi
        
        if [ "$auth_flag" = "auth" ]; then
            download_with_retry "$url" "$name" "$HF_TOKEN" || log_warning "Failed: $name"
        else
            download_with_retry "$url" "$name" || log_warning "Failed: $name"
        fi
    done
    
    ls -lah . | tee -a "$LOG_FILE"
fi

# ========== 最终验证 ==========
log "=== Phase 6: Final Verification ==="

cd "$WORKSPACE_DIR/$PROJECT_NAME"

python << 'EOF' 2>&1 | tee -a "$LOG_FILE"
import sys, torch, diffusers, transformers, accelerate, safetensors

print('=== Verification Report ===')
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {p.name} ({p.total_memory/1024**3:.1f}GB)')

packages = {
    'diffusers': diffusers,
    'transformers': transformers,
    'accelerate': accelerate,
    'safetensors': safetensors
}
for name, pkg in packages.items():
    print(f'{name}: {pkg.__version__}')

try:
    from diffusers import FluxPipeline
    print('✓ FluxPipeline available')
except: pass

try:
    from huggingface_hub import cached_download
    print('✓ cached_download available')
except: pass
EOF

# 创建激活脚本
cat > "$WORKSPACE_DIR/activate_env.sh" << 'EOF'
#!/bin/bash
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

cd /workspace
source sd-scripts-env/bin/activate
cd sd-scripts

echo "Environment activated. Run: python flux_train_network.py --help"
[ -d "/workspace/models" ] && ls -la /workspace/models/*.safetensors 2>/dev/null
EOF

chmod +x "$WORKSPACE_DIR/activate_env.sh"

# 清理trap
trap - EXIT

# 总结
END_TIME=$(date)
END_TIMESTAMP=$(date +%s)
DURATION=$((END_TIMESTAMP - START_TIMESTAMP))

log "=== Installation Summary ==="
log "Duration: $((DURATION/60))m $((DURATION%60))s"
log "Log file: $LOG_FILE"
log_success "Setup completed! Run: source $WORKSPACE_DIR/activate_env.sh" 

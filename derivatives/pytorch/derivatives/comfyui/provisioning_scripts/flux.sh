#!/bin/bash

set -e  # 脚本遇到错误直接退出，避免隐藏错误
set -o pipefail

source /venv/main/bin/activate
COMFYUI_DIR="${WORKSPACE}/ComfyUI"

# 预定义 APT/PIP 依赖（可扩展）
APT_PACKAGES=(
    "git" "ffmpeg" "libgl1" "libglib2.0-0"
)

PIP_PACKAGES=(
    "torch" "torchvision" "transformers"
)

# 自定义节点列表
NODES=(
    "https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/cubiq/ComfyUI_essentials"
    "https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes"
    "https://github.com/replicate/comfyui-replicate"
    "https://github.com/WASasquatch/was-node-suite-comfyui"
)

WORKFLOWS=(
    "https://gist.githubusercontent.com/robballantyne/f8cb692bdcd89c96c0bd1ec0c969d905/raw/2d969f732d7873f0e1ee23b2625b50f201c722a5/flux_dev_example.json"
)

CLIP_MODELS=(
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
)

UNET_MODELS=()
VAE_MODELS=()

### 核心流程 ###
function provisioning_start() {
    print_header

    update_comfyui
    install_apt_packages
    install_pip_packages
    install_custom_nodes

    # 下载 workflow 文件
    workflows_dir="${COMFYUI_DIR}/user/default/workflows"
    mkdir -p "${workflows_dir}"
    download_files "${workflows_dir}" "${WORKFLOWS[@]}"

    # 模型下载逻辑
    if validate_hf_token; then
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors")
    else
        UNET_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors")
        VAE_MODELS+=("https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors")
        sed -i 's/flux1-dev\.safetensors/flux1-schnell.safetensors/g' "${workflows_dir}/flux_dev_example.json" || true
    fi

    download_files "${COMFYUI_DIR}/models/unet" "${UNET_MODELS[@]}"
    download_files "${COMFYUI_DIR}/models/vae" "${VAE_MODELS[@]}"
    download_files "${COMFYUI_DIR}/models/clip" "${CLIP_MODELS[@]}"

    print_footer
}

### 更新 ComfyUI 主程序 ###
function update_comfyui() {
    echo "Updating ComfyUI core..."
    if [[ -d "${COMFYUI_DIR}/.git" ]]; then
        (cd "${COMFYUI_DIR}" && git pull)
    else
        echo "Warning: ${COMFYUI_DIR} not a git repo. Skipping update."
    fi
}

### 安装系统包 ###
function install_apt_packages() {
    if [[ ${#APT_PACKAGES[@]} -gt 0 ]]; then
        sudo apt-get update
        sudo apt-get install -y "${APT_PACKAGES[@]}"
    fi
}

### 安装 pip 包 ###
function install_pip_packages() {
    if [[ ${#PIP_PACKAGES[@]} -gt 0 ]]; then
        pip install --no-cache-dir "${PIP_PACKAGES[@]}"
    fi
}

### 安装自定义节点 ###
function install_custom_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        requirements="${path}/requirements.txt"
        if [[ -d "$path" ]]; then
            echo "Updating node: $repo"
            (cd "$path" && git pull)
        else
            echo "Cloning node: $repo"
            git clone --recursive "$repo" "$path"
        fi
        # 不论更新与否，都重新尝试安装依赖（更健壮）
        if [[ -f "$requirements" ]]; then
            pip install --no-cache-dir -r "$requirements" || true
        fi
    done
}

### 下载模型或文件 ###
function download_files() {
    local dir="$1"
    shift
    local urls=("$@")
    mkdir -p "$dir"
    echo "Downloading ${#urls[@]} file(s) to ${dir}..."
    for url in "${urls[@]}"; do
        echo "Downloading: $url"
        download_with_token "$url" "$dir"
    done
}

### 文件下载逻辑（支持 HF / Civitai Token）###
function download_with_token() {
    local url="$1"
    local dir="$2"
    local auth_token=""
    if [[ "$url" =~ huggingface\.co ]] && [[ -n "$HF_TOKEN" ]]; then
        auth_token="$HF_TOKEN"
    elif [[ "$url" =~ civitai\.com ]] && [[ -n "$CIVITAI_TOKEN" ]]; then
        auth_token="$CIVITAI_TOKEN"
    fi

    if [[ -n "$auth_token" ]]; then
        wget --header="Authorization: Bearer $auth_token" -c --content-disposition -P "$dir" "$url"
    else
        wget -c --content-disposition -P "$dir" "$url"
    fi
}

### 校验 Huggingface Token ###
function validate_hf_token() {
    if [[ -z "$HF_TOKEN" ]]; then return 1; fi
    response=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/whoami-v2")
    [[ "$response" == "200" ]]
}

### 格式化输出 ###
function print_header() {
    echo -e "\n==============================================="
    echo " Provisioning start: $(date)"
    echo "==============================================="
}

function print_footer() {
    echo -e "\n==============================================="
    echo " Provisioning complete: $(date)"
    echo "===============================================\n"
}

### 启动入口 ###
if [[ ! -f /.noprovisioning ]]; then
    provisioning_start
fi

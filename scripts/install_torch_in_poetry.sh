# get cuda version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -b 2-5)
# get python version
PYTHON_VERSION=$(python -c "import sys; print(sys.version_info[0])")
# get os
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

# select torch version
if [ "$CUDA_VERSION" = "10.2" ]; then
    TORCH_VERSION="1.12.1+cu102"
elif [ "$CUDA_VERSION" = "11.0" ]; then
    TORCH_VERSION="1.12.1+cu110"
elif [ "$CUDA_VERSION" = "11.1" ]; then
    TORCH_VERSION="1.12.1+cu111"
elif [ "$CUDA_VERSION" = "11.2" ]; then
    TORCH_VERSION="1.12.1+cu112"
elif [ "$CUDA_VERSION" = "11.3" ]; then
    TORCH_VERSION="1.12.1+cu113"
else
    echo "CUDA version not supported"
    exit 1
fi

# select torchvision version
if [ "$CUDA_VERSION" = "10.2" ]; then
    TORCHVISION_VERSION="0.13.1+cu102"
elif [ "$CUDA_VERSION" = "11.0" ]; then
    TORCHVISION_VERSION="0.13.1+cu110"
elif [ "$CUDA_VERSION" = "11.1" ]; then
    TORCHVISION_VERSION="0.13.1+cu111"
elif [ "$CUDA_VERSION" = "11.2" ]; then
    TORCHVISION_VERSION="0.13.1+cu112"
elif [ "$CUDA_VERSION" = "11.3" ]; then
    TORCHVISION_VERSION="0.13.1+cu113"
else
    echo "CUDA version not supported"
    exit 1
fi


# generate url to install pytorch, torchvision and torchaudio
TORCH_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION}/torch-${TORCH_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}m-${OS}_x86_64.whl"
TORCHVISION_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION}/torchvision-${TORCHVISION_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}m-${OS}_x86_64.whl"

# install pytorch, torchvision and torchaudio using poetry
poetry add ${TORCH_URL}
poetry add ${TORCHVISION_URL}

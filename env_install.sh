conda install \
    nvidia/label/cuda-12.1.0::cuda \
    nvidia/label/cuda-12.1.0::cuda-toolkit \
    nvidia/label/cuda-12.1.0::cuda-compiler \
    nvidia/label/cuda-12.1.0::cuda-libraries \
    -c nvidia/label/cuda-12.1.0 -c nvidia

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

bash ./env_install/5090_env_install.sh
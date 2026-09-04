# install cuda and cudatoolkit from conda
conda install -c nvidia cuda=12.8 -y 
conda install -c nvidia cuda-toolkit=12.8 -y 
conda install -c nvidia cuda-nvcc=12.8 -y 

# install gcc and g++ version 10
conda install -c conda-forge gxx_linux-64=10 gcc_linux-64=10 -y 

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install -y numpy==1.26.4
pip install "scipy<=1.15.3" --no-cache-dir --force-reinstall

pip install warp-lang
pip install usd-core matplotlib
pip install "pyglet<2"
pip install open3d
pip install trimesh
pip install rtree 
pip install pyrender

pip install stannum
pip install termcolor
pip install fvcore
pip install wandb
pip install moviepy imageio
conda install -y opencv
pip install cma
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation --no-cache-dir

# Install the env for realsense camera
pip install Cython
pip install pyrealsense2
pip install atomics
pip install pynput

# Install the env for image upscaler using SDXL
pip install diffusers
pip install accelerate

pip install gsplat==1.4.0
pip install kornia
pip install h5py
pip install einops
pip install tensorboard

# set up gcc and g++ path
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

cd src/gaussian_splatting/
pip install submodules/diff-gaussian-rasterization/ --no-build-isolation
pip install submodules/simple-knn/ --no-build-isolation
cd ../..

pip install torch-scatter torch-geometric \
    -f https://data.pyg.org/whl/torch-2.8.0+cu128.html \
    --no-cache-dir

conda install -c conda-forge libegl libglu -y
pip install plyfile

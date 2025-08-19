# depthnav

## Setup
```bash
git clone git@github.com:rislab/depthnav.git --init --recursive
cd depthnav/
python3.9 -m venv .venv # install python3.9 if you do not have it on the machine
source .venv/bin/activate
pip install --upgrade pip

# system-wide dependencies
sudo apt-get update
sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
sudo apt-get install -y libcgal-dev

# pip installable dependencies
pip install -r requirements.txt

# verify that torch is cuda enabled
python -c "import torch; print(torch.__version__); print(torch.cuda.get_arch_list()); print(torch.randn(1).cuda())"
# you should see an output like:
# 2.2.1+cu121
# ['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
# tensor([0.6627], device='cuda:0')

# habitat-sim
cd habitat-sim
python setup.py install --with-cuda --build-type Release --cmake-args="-DPYTHON_EXECUTABLE=$(which python) -DCMAKE_CXX_FLAGS_RELEASE='-Ofast -march=native'"
```

`depthnav/` contains the core infrastructure and classes for rendering,
dynamics, and environment. `examples/` contains scripts the create class
instances and are used to train the policy.

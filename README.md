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

The environments need to be pre-generated before training so we can cache the geodesic fields.

The dataset (includes assets, scene config files, geodesics) are located under
`VisFly-datasets/datasets/skinny_dataset/`.

Generate the geodesics: Edit the `generate_training_envs.py` script by changing the `if True:` to `if False:` in order to disable generating geodesics. For example, if you only want to train on `level1` you can disable generating the geodesics for all the other levels.

```bash
source .venv/bin/activate
cd VisFly/skinny_examples/geodesics
python generate_training_envs.py
```

## Training

Train a policy using only level1 of the curriculum
```bash
python skinny_examples/navigation2/run_nav2_level1.py
```

There are some other training scripts in `skinny_examples/navigation2/` with
different configs, such as `run_nav2_curriculum.py` which trains for more
iterations over 4 levels of different envrionments.

## Evaluating

To evaluate the policy, you can run the `eval_visual.py` script which will run
a batch of `num_envs` agents visually.

```bash
python skinny_examples/navigation2/eval_visual.py \
    --cfg_file "skinny_examples/navigation2/eval_cfg/nav2_ring.yaml" \
    --weight "skinny_examples/navigation2/logs/level1/level1_body_1.pth" \
    --policy_cfg_file "skinny_examples/navigation2/logs/level1/level1_body.yaml" \
    --render \
    --num_envs 4 \
    --plot 1 
```

You can also run `eval_logger.py` which will run without rendering and will save
the rollout stats to a csv.
```bash
python skinny_VisFly/scripts/eval_logger.py \
    --cfg_file "skinny_examples/navigation2/eval_cfg/nav2_ring.yaml" \
    --weight "skinny_examples/navigation2/logs/level1/level1_body_1.pth" \
    --policy_cfg_file "skinny_examples/navigation2/logs/level1/level1_body.yaml" \
    --num_envs 4
```


## Other Scripts

Visualize the training environments with geodesic overlay
```bash
python skinny_examples/geodesics/interactive_flow_field.py --scene level_1/ring_walls_small
```
# Prerequisites
- Linux
- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

# Setup ENV
## dataset (required, on host)
 cd 0.peft/dataset && unzip -o sft_data.zip

> Extracts `train.jsonl` (35k rows). The recipe expects it at `dataset/train.jsonl`. The `.jsonl` is git-ignored; only the `.zip` is versioned.

## pull image
 docker pull nvcr.io/nvidia/nemo-automodel:26.04.00

## toolkit (one-time)
 install nvidia-container-toolkit
 sudo nvidia-ctk runtime configure --runtime=docker
 sudo systemctl restart docker
 docker info | grep -i runtime

## run container (from repo root)
 docker run --gpus all -it --rm --shm-size=8g -v $(pwd)/0.peft:/tmp/0.peft/ nvcr.io/nvidia/nemo-automodel:26.04.00

> Mounts host `0.peft` to `/tmp/0.peft`. Run everything below from `/tmp/0.peft` inside the container.

# HF login [optional]
 hf auth login
 hf auth whoami

# Run PEFT (inside container)
 cd /tmp/0.peft
 automodel recipe.yaml

## low-VRAM verify (e.g. 20 GB card)
 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True automodel recipe.smoke.yaml

# Test checkpoint (inside container)
 python infer.py --num-samples 10 --show-expected
 python infer.py --question "What is 12*13?"

# Files
- `recipe.yaml` — full training recipe
- `recipe.smoke.yaml` — low-VRAM verification recipe
- `mydataset.py` — chat-format SFT dataset, loss on assistant tokens only
- `infer.py` — checkpoint tester: generation + format PASS/FAIL
- `smoke_test_dataset.py` — dataloader check, no GPU needed
- `fix_encoding.py` — UTF-8 repair for externally sourced JSONL (stock data is clean)
- `dataset/sft_data.zip` — `train.jsonl`, 35k rows

## [Parameter Tuning Guide](TUNING.md)

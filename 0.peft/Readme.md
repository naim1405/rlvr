# Prerequisites
- Linux
- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

# Setup ENV
## pull image
 docker pull nvcr.io/nvidia/nemo-automodel:26.04.00

## install toolkit
 install nvidia-container-toolkit

## configure toolkit
 sudo nvidia-ctk runtime configure --runtime=docker

## restart docker
 sudo systemctl restart docker

## verify docker runtime
 docker info | grep -i runtime
 Runtimes: io.containerd.runc.v2 nvidia runc
 Default Runtime: runc

## run docker container
 docker run --gpus all -it --rm --shm-size=8g -v $(pwd)/0.peft:/tmp/0.peft/ nvcr.io/nvidia/nemo-automodel:26.04.00

> The above command will start the docker container and mount the local directory `0.peft` to `/tmp/0.peft` inside the container. `automodel` will need to be run from /tmp/0.peft inside the container.


# Run PEFT
 cd /tmp/checkpoints
 automodel recipe.yaml

# huggingface login [optional]
 hf auth login
token: __TOKEN__
 hf auth whoami


## [Parameter Tuning Guide](Tuning.md)



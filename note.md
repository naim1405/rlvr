# Setup ENV
## pull image
 docker pull nvcr.io/nvidia/nemo-automodel:26.04.00

## install toolkit
 install nvidia-container toolkit

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


# huggingface login [optional]
 hf auth login
token: __TOKEN__
 hf auth whoami



# Run PEFT
 cd /tmp/checkpoints
 automodel recipe.yaml

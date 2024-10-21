# dynamic_batching_optimization

Updates    : 2024.10.16

Author     : Seongho Kim

Email      : [seongho-kim@yonsei.ac.kr](https://seongho-git.github.io/)

Github     : seongho-git

The concept of this repository is based on the paper titled [Survey and Evaluation of Converging Architecture in LLMs Based on the Footsteps of Operations](https://arxiv.org/abs/2410.11381).

The codes will be updated in November.


## Installation
```bash
https://github.com/seongho-git/dynamic_batching_optimization.git
```

### Docker Setting (for Jetson Orin 32GB)

### Install with clean base (for GPUs)
```bash
sudo docker run --runtime nvidia -it --privileged --name phi3 -v {MOUNT_PATH}:/mnt --network=host dustynv/jetson-inference:r36.3.0
```
```bash
cd && git clone https://github.com/seongho-git/dynamic_batching_optimization && cd dynamic_batching_optimization && pip -r install requirements.txt
```

The **MOUNT_PATH** should be a directory that contains the following files and folders, with the same names as listed below, and it must be mounted to the /mnt folder:

```bash
# in docker container,
~/ # /root/
- yphi.py
- requirements.txt

${MOUNT_PATH}
- Phi-3-medium-4k-instruct # Phi3 model (from huggingface)
- test_dataset.jsonl # dataset
```
## Run
Within the Docker container, you can run the dataset using the yphi.py file located in the ~/ (root folder).

You can check the results by executing the following commands:
```bash
cd && python3 yphi.py
```

## Download Phi3 model (optional)

```bash
git lfs install && git lfs clone https://huggingface.co/microsoft/Phi-3-medium-4k-instruct
```

# VERL_VLM
The vlm version of verl, which currently only supports Qwen 2.5-VL.

# Installation
```bash
# Create the conda environment
conda create -n verl python==3.10
conda activate verl
pip3 install -e .

# Install vLLM>=0.7
pip3 uninstall vllm
pip3 install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Install flash-attn
pip3 install --use-pep517 flash-attn --no-build-isolation

pip3 uninstall torch
# Support CUDA 12.1
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install qwen_vl_utils
```

# RUN
```bash
python prepare_data.py
sh run_qwen2.5-vl-3b.sh
```

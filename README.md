# EPIC 2.0

## Installation

### Requirements

- Python 3.11
- CUDA 12.8
- PyTorch 2.8.0

### Setup

```bash
conda env create -f environment.yml
conda activate EPIC
```

## Usage

### Structured Pruning

#### Basic Pruning

```bash
python -m pruning.cli \
    --model DeepLabV3Plus_resnet \
    --pruning-ratio 0.5 \
    --strategy magnitude \
```

#### Sensitivity Analysis Prunning

```bash
python -m pruning.cli \
    --model DeepLabV3Plus_resnet \
    --pruning-ratio 0.5 \
    --strategy magnitude_taylor \
```

#### Advanced Pruning with KD-Lite

```bash
python -m pruning.cli \
    --model UNET_efficientnet \
    --pruning-ratio 0.7 \
    --strategy magnitude_taylor \
    --enable-kd \
    --kd-temperature 4.0 \
    --kd-alpha 0.7 \
    --steps 10 \
    --fine-tune-epochs 20
```

### Evaluation

```bash
python pruning/eval/run_eval.py
```

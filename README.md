# remote_aurora

This repository is the server-deployment mirror for AuroraIG + ReconVLA experiments.

For the detailed Chinese server deployment notes, see:

```text
aurora/docs-main/server_deployment.md
```

## Current Deployment Workflow

The active training/debugging environment is on the HPC server. The original code and some upstream repositories are private or need local adaptation, while the server cannot reliably access GitHub directly. To make code changes reproducible:

1. Edit code locally in:

   ```text
   C:\Users\29424\Desktop\华科\remote_aurora
   ```

2. Commit and push local changes to the public GitHub mirror:

   ```text
   https://github.com/supa-1/remote_aurora.git
   ```

3. Pull the public mirror on the server. If direct GitHub access fails, use `ghfast.top`:

   ```bash
   cd /home/share/ltwwa4al/home/Huangjian_Hust/text_recon
   git pull https://ghfast.top/https://github.com/supa-1/remote_aurora.git master
   ```

Model weights, checkpoints, processed CALVIN data, JSONL outputs, and cache files are intentionally ignored by `.gitignore`; keep them on the server or transfer them separately.

## Server Layout

The current server layout is expected to be:

```text
/home/share/ltwwa4al/home/Huangjian_Hust/text_recon/
├── aurora/
├── ReconVLA/
└── calvin/
```

Processed datasets should be written under:

```text
/home/share/ltwwa4al/home/Huangjian_Hust/text_recon/calvin/dataset/process/<DATASET_NAME>/
```

For example:

```text
calvin/dataset/process/calvin_debug_dataset/
├── processed_json/
├── processed_images/
└── consistency_pairs/
```

## ARM / HPC Adaptation

The server is ARM/aarch64, so x86 CUDA/PyTorch wheels and some packages cannot be assumed to work. The scripts under this repository have been partially adapted for the server:

- `aurora/scripts/train_vla/hpc_env.sh` loads the HPC conda and module environments.
- `aurora/scripts/train_vla/a100_lora_smoke.sh` runs A100 LoRA smoke training with `BIT=16` by default to avoid `bitsandbytes` on ARM.
- `aurora/scripts/build_llm_pairs_calvin.sh` uses the local Qwen-8B path, disables 4-bit loading, uses eager attention for LLM generation, and disables Qwen thinking output for fake-instruction generation.

The server environment currently needs the following setup before training scripts run:

```bash
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
conda activate reconvla

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module load compilers/nvhpc_sdk/23.5_cuda_11.8_12.1
module load libs/cudnn/8.9.5_cuda12
```

For fake-instruction generation, use the `aurora` conda environment instead of `reconvla`.

## DSUB Job Scripts

The server should run training through submitted job scripts rather than only interactive shell commands. A 100-step consistency LoRA smoke job can be written as:

```bash
#!/bin/bash
#DSUB -n aurora_consistency_100step
#DSUB -N 1
#DSUB -A root.ltwwa4al
#DSUB -R "cpu=8;gpu=1;mem=80000"
#DSUB -oo %J.out
#DSUB -eo %J.err

set -euo pipefail

source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
conda activate reconvla

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module load compilers/nvhpc_sdk/23.5_cuda_11.8_12.1
module load libs/cudnn/8.9.5_cuda12

cd /home/share/ltwwa4al/home/Huangjian_Hust/text_recon/aurora

LORA_ENABLE=True \
BIT=16 \
MAX_STEPS=100 \
ENABLE_CONSISTENCY_AUX=True \
CONSISTENCY_AUX_WEIGHT=0.3 \
OUTPUT_DIR=checkpoints/a100_lora_consistency_100step \
bash scripts/train_vla/a100_lora_smoke.sh
```

Submit with:

```bash
dsub -s text_recon/a100_lora_100step.sh
```

If the job stays `PENDING` with a resource-quota message, reduce CPU/RAM first, for example `cpu=4;gpu=1;mem=64000`. The `mem` value is system RAM, not GPU VRAM.

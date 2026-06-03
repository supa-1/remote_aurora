# 服务器部署与同步说明

本文档记录当前 AuroraIG + ReconVLA 在服务器上的部署方式、代码同步方式和 HPC 运行注意事项。

## 1. 代码同步方式

当前主要工作环境在服务器上，但由于以下限制，需要通过本地 Git 仓库中转管理代码：

- 现有原始 GitHub 仓库包含 private 内容，部分脚本还需要针对服务器路径和环境做适配修改。
- 服务器无法稳定直连 GitHub，经常需要使用 `ghfast.top` 加速。
- 因此在本地创建了一个用于管理服务器代码的镜像仓库：

```text
C:\Users\29424\Desktop\华科\remote_aurora
```

当前推荐工作流：

1. 在本地 `remote_aurora` 中修改代码。
2. 提交并 push 到公开 GitHub 镜像：

   ```text
   https://github.com/supa-1/remote_aurora.git
   ```

3. 在服务器上拉取更新。若直连 GitHub 不稳定，使用 `ghfast.top`：

   ```bash
   cd /home/share/ltwwa4al/home/Huangjian_Hust/text_recon
   git pull https://ghfast.top/https://github.com/supa-1/remote_aurora.git master
   ```

模型文件、checkpoint、处理后的 CALVIN 数据、JSONL 输出和缓存文件不进入 Git，由 `.gitignore` 忽略；这些文件保留在服务器或单独传输。

## 2. 服务器目录约定

服务器上的当前目录结构应保持为：

```text
/home/share/ltwwa4al/home/Huangjian_Hust/text_recon/
├── aurora/
├── ReconVLA/
└── calvin/
```

处理后的数据统一放到：

```text
/home/share/ltwwa4al/home/Huangjian_Hust/text_recon/calvin/dataset/process/<DATASET_NAME>/
```

例如 `calvin_debug_dataset` 对应：

```text
calvin/dataset/process/calvin_debug_dataset/
├── processed_json/
├── processed_images/
└── consistency_pairs/
```

`calvin_task_d_d` 等正式数据集也应按同样规则放在 `process/calvin_task_d_d/` 下，便于脚本通过 `DATASET_NAME` 切换。

## 3. ARM / HPC 适配

服务器是 ARM/aarch64 架构，因此不能默认使用 x86 环境下的 CUDA / PyTorch / bitsandbytes wheel。当前仓库中的脚本已经做了部分适配：

- `aurora/scripts/train_vla/hpc_env.sh`
  - 加载 HPC conda 和 module 环境。
  - 加载 CUDA/NVHPC/cuDNN 模块。
  - 处理部分 ARM 环境下的动态库加载问题。
- `aurora/scripts/train_vla/a100_lora_smoke.sh`
  - 面向 A100 单卡 LoRA smoke 训练。
  - 默认 `BIT=16`，避免 ARM 上的 `bitsandbytes` 兼容问题。
- `aurora/scripts/build_llm_pairs_calvin.sh`
  - 默认使用 `aurora/models/qwen-8b`。
  - 禁用 4bit。
  - LLM 生成使用 `eager` attention。
  - 默认禁用 Qwen3 thinking 输出，避免 `<think>` 和解释文本污染假指令。

训练环境一般需要：

```bash
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
conda activate reconvla

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module load compilers/nvhpc_sdk/23.5_cuda_11.8_12.1
module load libs/cudnn/8.9.5_cuda12
```

生成真假指令时使用 `aurora` conda 环境；训练 ReconVLA 时使用 `reconvla` conda 环境。

## 4. 作业脚本要求

服务器实际运行训练时应编写 DSUB 作业脚本提交，而不是长期依赖交互式 shell。示例 100-step 一致性辅助 LoRA 作业如下：

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

提交示例：

```bash
dsub -s text_recon/a100_lora_100step.sh
```

如果作业长期 `PENDING`，用以下命令查看原因：

```bash
djob -L <JOB_ID>
```

若提示资源配额不满足，例如 `Resource quota assigned ... cannot satisfy resource requirement`，优先降低 CPU/RAM 申请：

```text
#DSUB -R "cpu=4;gpu=1;mem=64000"
```

其中 `mem` 是节点系统内存/RAM，不是 GPU 显存。GPU 显存由调度到的 GPU 型号决定。

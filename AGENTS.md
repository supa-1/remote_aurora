# AGENTS.md

默认允许子代理。

## 项目上下文

当前仓库是 AuroraIG + ReconVLA 在服务器部署时使用的代码镜像。实际训练和数据生成主要发生在服务器：

```text
/home/share/ltwwa4al/home/Huangjian_Hust/text_recon
```

本地维护仓库路径：

```text
C:\Users\29424\Desktop\华科\remote_aurora
```

本地修改应先提交并 push 到公开 GitHub 镜像：

```text
https://github.com/supa-1/remote_aurora.git
```

服务器再 pull。服务器无法稳定直连 GitHub 时，使用：

```bash
git pull https://ghfast.top/https://github.com/supa-1/remote_aurora.git master
```

## 服务器与脚本约束

- 服务器是 ARM/aarch64 架构，不能默认使用 x86 的 PyTorch、CUDA、bitsandbytes wheel。
- 现有脚本已经做了部分 HPC 适配，尤其是：
  - `aurora/scripts/train_vla/hpc_env.sh`
  - `aurora/scripts/train_vla/a100_lora_smoke.sh`
  - `aurora/scripts/build_llm_pairs_calvin.sh`
- ReconVLA 训练环境使用 `reconvla` conda 环境。
- AuroraIG 真假指令生成使用 `aurora` conda 环境。
- 训练优先使用 `BIT=16`，避免 ARM 上的 bitsandbytes 问题。
- Qwen3 真假指令生成默认禁用 thinking，避免 `<think>` 或解释性文本进入 JSONL。

## 数据与模型

不要把以下内容提交到 Git：

- 模型权重
- checkpoint
- HuggingFace/cache 文件
- processed CALVIN 数据
- consistency pair JSONL
- wandb/runs/logs

这些内容应保留在服务器或单独传输，仓库通过 `.gitignore` 忽略。

## 作业提交

服务器实际运行训练时需要编写 DSUB 作业脚本，不要只依赖交互式命令。作业脚本示例和资源申请说明见：

```text
aurora/docs-main/server_deployment.md
```

注意：`#DSUB -R` 里的 `mem` 是节点系统内存/RAM，不是 GPU 显存。GPU 显存由申请到的 GPU 型号决定。

@C:\Users\29424\.codex\RTK.md

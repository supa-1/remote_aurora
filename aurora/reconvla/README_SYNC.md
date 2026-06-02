# Reconvla 同步说明

本目录用于在 AuroraIG 内本地运行 Reconvla 训练主链路。

## 来源
- 源目录：`/home/supa1/myreconvla/Reconvla/reconvla`
- 同步内容：`train_vla.py`、`action_tokenizer.py`、`pre_train_vla_action.py`、`recon/`、`statistics.yaml`、`model_parameters.json`

## 约定
- 同步来的文件不在文件名中添加 `copy`。
- 在关键入口文件内部增加来源说明，便于审计与后续对齐。
- 若上游更新，请重新同步并做差异检查。

## 与 AuroraIG 关系
- `reconvla/` 提供基础训练链路。
- `auroraig/` 提供你的增量创新模块（对比学习、伪指令增强）。

## 当前本地增量（相对上游）
- 在 `train_vla.py` 增加一致性辅助字段读取（含 fake pool 与类型权重参数）。
- 在 `recon/model/language_model/recon_qwen.py` 增加样本级一致性损失加权输入。

建议：每次与上游同步后，重点回看上述两个文件的冲突与行为是否被覆盖。

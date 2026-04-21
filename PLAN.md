# PLAN

## 用户约束
- 仅使用 CUDA_VISIBLE_DEVICES=0。
- 优先使用 conda prefix: /home/xiaoql26/dtank_disk01/xiaoql26.from.denali-3/xiaoql26/lxq_new/envs/mamba4d_clean。
- 先跑 baseline，再做 strict-causal + training-only PSD + anti-collapse 的最小侵入实现。
- 先做小型验证；仅当结果正面时再启动大型实验。
- 不破坏原 baseline；新增模块/脚本/配置优先。

## 当前状态
1. 已审计 repo / 环境 / 数据 / entrypoints。
2. 已跑通 MSR recognition baseline、causal-only、causal+PSD 小型实验。
3. 已完成 strict-causal backbone，当前是 strict-causal full-sequence forward，不是部署级 streaming cache。
4. 已加入 summary extractor + multi-horizon delta predictor + var-cov / SIGReg anti-collapse regularizer，auxiliary branch 只在训练期执行。
5. causality sanity check 已通过：扰动未来帧后，当前步 logits 最大差异为 0。
6. 小型结果支持进入 full-data 对照；full-data 上 causal+PSD 与 causal-only 打平，因此暂停追加更大实验。

## 主要风险
- pointnet2 / knn_cuda / mamba_ssm 自定义依赖兼容性。
- 原始 temporal embedding 存在 future leakage，需要最小改造以保持可运行。
- 原仓库无配置系统，需以新增脚本/参数方式暴露超参。

## 下一步建议
- 优先调低 `lambda_reg` / `lambda_div` 或做 auxiliary warmup。
- 比较 horizon `[1]`、`[1, 2]`、`[1, 2, 4]`，先避免长 horizon 对分类早期学习的干扰。
- 如果继续使用 SIGReg，需要修正当前最小奇异值长期贴近 floor 的问题，再进入 full-data。

## Round 2 Summary

本轮聚焦 “summary source / target definition 是否 task-aligned”，不做大 sweep。

已完成：
- 现有 full checkpoint probe，确认旧 summary 的确更可预测但更弱判别。
- 在 `summary_extractor.py` 中保留 `pooled_mlp`，新增 `token_query`。
- 加入 train-only semantic anchor 和统一 warmup。
- 跑完四组对照：
  1. `pooled_mlp` 无 semantic
  2. `pooled_mlp` + semantic
  3. `token_query` 无 semantic
  4. `token_query` + semantic
- 对 A / D 再做 probe。

当前判断：
- `token_query` 本身不是直接收益来源；没有 semantic anchor 时它明显掉点且 slot 重复严重。
- semantic anchor 对 `token_query` 有补救作用，但并没有把它推到超过旧 summary 的水平。
- 因此本轮不继续 full-data；下一步更值得做的是重新设计 query summary 的 slot diversity / shared-query 机制，或引入更强的 task-conditioned summary target，而不是先扫 horizon 或 SIGReg。

## Round 3 Plan: Task-Aligned Target Verification

本轮固定主线：
- 继续只做 MSR strict-causal recognition，不切回 HOI4D 主实验。
- `summary_mode` 固定为 `pooled_mlp`。
- regularizer 固定 `var_cov`，horizon 固定 `[1]`。
- 统一使用轻量 auxiliary warm-up，不做 schedule sweep。
- `token_query` 不再作为主实验对象。

本轮执行顺序：
1. 复用现有 `full_causal_psd` 与 `round2_old_no_sem_pilot` checkpoint，重跑低成本 probe，确认 “predictable but task-misaligned”。
2. 最小侵入实现三组：
   - A: `summary -> future summary delta`
   - B: A + `soft_main_kl` semantic alignment
   - C: `summary -> future projected pooled-state delta`
3. 在与上一轮一致的小预算下跑 A/B/C pilot。
4. 对 A/B/C checkpoint 跑 probe，比对 summary 判别性与 target predictability。
5. 只有当 C 明显优于 A，或 B/C 呈现清晰正向趋势并且 probe 显示 task-alignment 改善时，才进入 1-seed full-data。

Round 3 执行结果：
- low-cost probe 复核后，full old PSD 仍表现为 `summary delta` 显著比 `pooled delta` 更可预测（0.0036 vs 0.2058），但 `summary` 标签 probe 仍略低于 `pooled`（0.7441 vs 0.7576），继续支持 “predictable but not sufficiently task-aligned”。
- pilot 结果：
  - A `summary_delta`: `val_video_acc=0.4267`
  - B `summary_delta + soft_main_kl`: `0.3879`
  - C `projected_pooled_target_delta`: `0.4052`
- pilot probe 结果：
  - A: `summary_probe=0.3793`, `pooled_probe=0.4483`
  - B: `summary_probe=0.4095`, 但训练中 `loss_sem` 升到 `2.03`，主任务下滑
  - C: `summary_probe=0.4267`, 已逼近并轻微超过 `pooled_probe=0.4181`，同时 `projected_pooled_target_delta` 仍可预测（0.0130）
- 因此按用户门槛进入 1-seed full-data，只放大 C，不重开新变量。
- full-data C 最终达到 `val_video_acc=0.7811`，超过现有 `full_causal=0.7677` 与 `full_old_psd=0.7677`，且 causality check 继续为 0 差异。

## Round 4 Outcome: Matched Full A vs C

本轮已按 matched full-data 规范重跑 A/C，并补了一个 matched causal-only：
- A `summary_delta`: `best val_video_acc=0.7879`
- C `projected_pooled_target_delta`: `best val_video_acc=0.7811`
- causal-only: `best val_video_acc=0.7946`

已确认 A/C 配置层面的唯一差异是：
- `model.psd.pred_target_type`

Round 4 结论：
- matched 单 seed full-data 不支持 “C 比 A 更强” 的强 claim。
- C 的 summary probe 确实更强，但主任务没有赢过 A，更没有赢过 causal-only。
- 下一步不做新 target / HOI4D，优先补 2~3 seed 的 matched A/C，判断 C 的优势是否仅停留在 representation alignment。

## Round 5 Plan: Multi-Seed Matched A vs C

目标：
- 以现有 round-4 `seed=0` 结果为第一颗 seed。
- 新增 `seed=1` 与 `seed=2` 的 matched A/C full-data run。
- 保持完全相同的训练预算、数据、warm-up、summary source、regularizer 与 eval protocol。
- 只允许 `seed` 与 `pred_target_type` 变化。

执行顺序：
1. 复制 round-4 A/C full config，生成 `seed=1` 与 `seed=2` 四个配置。
2. 用两张卡并行发车同一 seed 的 A/C：
   - `seed=1`: A on GPU0, C on GPU1
   - `seed=2`: A on GPU0, C on GPU1
3. 对 `seed=1/2` 的 best checkpoint 跑 probe。
4. 聚合 `seed=0/1/2` 的 A/C 主任务、probe、predictability 统计。
5. 输出 3-seed mean/std、paired delta 与论文级表图。

成功判据：
- 若 C 在多 seed 下稳定提高 `val_video_acc`，再考虑是否将 task-aligned target 迁移到 HOI4D。
- 若 C 只稳定提高 summary probe / alignment，而主任务均值仍不优于 A，则论文 claim 收缩到 “representation alignment improved, but recognition gain is not yet robust on MSR”。

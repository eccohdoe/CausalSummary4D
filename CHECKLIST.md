# CHECKLIST

- [x] 确认环境类型（conda prefix / venv）
- [x] 定位训练入口、模型入口、数据入口
- [x] 跑通 baseline smoke
- [x] 确认 baseline 日志/指标输出路径
- [x] 实现 strict-causal 最小版本
- [x] 实现 PSD summary / predictor / regularizer
- [x] 运行 causality sanity check
- [x] 运行小型 baseline 对比实验
- [x] 根据结果决定是否启动大型实验
- [x] 运行 full-data baseline / causal / causal+PSD 对照
- [x] 汇总 diff / 命令 / 结果 / 建议

## 结论状态

- 小型实验正面：baseline 0.3362，causal-only 0.3793，causal+PSD var-cov 0.3922，causal+PSD SIGReg 0.4009。
- large 对照已完成：baseline 0.8485，causal-only 0.7677，causal+PSD 0.7677。
- causality sanity check 全部通过，future perturbation 的当前步 logits 差异为 0。
- 不建议继续追加更大实验；下一步应先调低/调度 auxiliary regularization 或缩短 horizon。

## Round 2: Summary Alignment

- [x] 用现有 full checkpoint 做 pooled state vs summary probe
- [x] 实现 `summary_mode=pooled_mlp/token_query` 双路径
- [x] 实现轻量 semantic anchor 与 `lambda_sem`
- [x] 实现统一 auxiliary warmup
- [x] 跑完 A/B/C/D 四组 pilot
- [x] 跑完 A vs D follow-up probe
- [x] 做出是否进入 full-data 的决定

### Round 2 结论

- 旧 full checkpoint probe 支持 “summary 更可预测但不够判别”：
  - full pooled delta probe MLP MSE 0.2047
  - full summary delta probe MLP MSE 0.0037
  - full pooled label probe MLP video acc 0.7576
  - full summary label probe MLP video acc 0.7374
- 四组 pilot：
  - A `old_no_sem`: 0.4181
  - B `old_sem`: 0.3836
  - C `query_no_sem`: 0.3793
  - D `query_sem`: 0.4181
- `token_query` 需要 semantic anchor 才能追回性能，但仍未超过旧 summary 路径。
- `query_sem` probe 仍显示 summary 比 pooled state 更难分类：
  - pooled label probe MLP video acc 0.4397
  - summary label probe MLP video acc 0.3534
- 本轮不进入 full-data。

## Round 3: Task-Aligned Target

- [x] 复查 full / old-no-sem checkpoint，确认 task-misalignment 仍是主问题
- [x] 实现 A `summary_delta`
- [x] 实现 B `summary_delta + soft_main_kl`
- [x] 实现 C `projected_pooled_target_delta`
- [x] 跑完 A/B/C pilot
- [x] 跑完 A/B/C probe
- [x] 判断是否进入 1-seed full-data
- [x] 跑完 1-seed full-data C
- [x] 跑完 full-data C probe

### Round 3 结论

- low-cost probe 继续支持旧 PSD 的核心问题不是 “不可预测”，而是 “predictive target 不够贴近主任务”：
  - full old PSD: `pooled_probe=0.7576`, `summary_probe=0.7441`
  - full old PSD: `pooled_delta=0.2058`, `summary_delta=0.0036`
- pilot A/B/C：
  - A `summary_delta`: `0.4267`
  - B `summary_delta + soft_main_kl`: `0.3879`
  - C `projected_pooled_target_delta`: `0.4052`
- B 没有成为有效方向：soft semantic alignment 虽把 summary probe 提到 `0.4095`，但 `loss_sem` 膨胀到 `2.03`，主任务比 A 更差。
- C 是本轮真正的正向信号：
  - pilot `summary_probe=0.4267`，已优于 `pooled_probe=0.4181`
  - full `summary_probe=0.7845`，超过 `pooled_probe=0.7576`
  - full `val_video_acc=0.7811`，超过现有 `full_causal=0.7677` 与 `full_old_psd=0.7677`
  - causality check 全程通过，`max_abs_logit_diff=0.0`

## Round 4: Matched Full A vs C

- [x] 生成 matched full A 配置
- [x] 生成 matched full C 配置
- [x] 跑完 matched full A
- [x] 跑完 matched full C
- [x] 跑完 matched full causal-only
- [x] 跑完 matched full A/C probe
- [x] 生成 round-4 表格与图

### Round 4 结论

- matched `seed=0` full-data：
  - causal-only: `0.7946`
  - A: `0.7879`
  - C: `0.7811`
- A/C 完全 matched，唯一差异是 `pred_target_type`。
- C 相比 A 的确缩小了 `summary` 与 `pooled` 的 probe gap，但没有换来更高的主任务精度。
- 因此已进入 round-5，多 seed 复核是当前唯一高价值下一步。

## Round 5: Multi-Seed Matched A vs C

- [x] 固定使用 round-4 `seed=0` 结果作为第一颗 seed
- [x] 生成 `seed=1` matched A/C 配置
- [x] 生成 `seed=2` matched A/C 配置
- [x] 跑完 `seed=1` matched A/C
- [x] 跑完 `seed=2` matched A/C
- [x] 跑完 `seed=1/2` probe
- [x] 聚合 `seed=0/1/2` A/C mean/std 与 paired delta
- [x] 生成 round-5 多 seed 表格与图
- [x] 给出 claim 是否收缩到 “representation alignment” 的最终判断

### Round 5 结论

- 3-seed matched full-data 主结果：
  - A best `77.89 ± 0.84`
  - C best `78.79 ± 0.95`
  - `C - A = +0.90` 个点（paired mean）
- 3-seed final acc：
  - A final `76.09 ± 0.27`
  - C final `77.78 ± 0.99`
- 3-seed summary probe：
  - A `72.50 ± 1.24`
  - C `76.99 ± 3.17`
  - `C - A = +4.49` 个点（paired mean）
- seed-level paired delta：
  - seed0: `C - A = -0.67`
  - seed1: `+3.37`
  - seed2: `0.00`
- 结论不再是 “只有 alignment 没有 recognition gain”。
  当前证据更支持：
  - task-aligned target 稳定提升 summary 判别性；
  - 在 3-seed matched MSR full-data 上，也带来正向平均主任务收益；
  - 但 gain 仍属中等幅度，论文里仍应保留 “multi-seed but small-scale” 的谨慎表述。

# 原 M8 最终审核记录

2026-11-20 核对。当前公开入口 `from experiments.workflow_extensions.m8 import load_original, run` 加载原 BUN/Cr 第二轮冻结计划。后续加权、统一刷新等研究候选均需显式传入相应计划。无参数运行模块 CLI 会执行后续 study 回测；部署调用方式见 [README](README.md)。

| 核对项 | 结果 |
|---|---|
| 默认配置 | original_bun.json 与原实验归档 frozen.json 逐字节一致，SHA-256 为 `7be92ebf88a27c6aff96df51fbb2728dbddaf641f4b0b5577e0ca9491cccc3cf` |
| 协议与参考患者 | original_protocol.json 与原归档 protocol.json 逐字节一致，SHA-256 为 `6a1b9d31247ec19b8133152ce488bf07fbf2b776b6a11e599715068d5dbae25b`。加载器验证数据文件哈希；48 个互异参考 ID 全部位于训练划分，按协议顺序重建 Context |
| 全预算路由 | 从两个密度组各取一条真实轨迹，54 个预算 × 2 组共 108 个组合，M8 与归档代码选出的 Branch 逐项相等 |
| 付费时点、缓存和支出 | 将旧回放抽到的各根格点作为相同输入送入新部署流程，108 个组合付费时点一致、费用精确加和为传入 B，40 轮服务缓存最大绝对差为 1.11×10⁻¹⁶ |
| 报告数值 | 从原 summary.csv 重读 route 的四预算平均 MAE，全体 5.711750，高波动层 7.270082，低波动层 3.713811，与报告舍入值一致 |

一致性测试控制了发布格点，检查的是给定相同观测后的行为。历史实验用浮点核配对回放，现部署使用同一理论指数核的精确随机整数采样，并以有理数分配费用。两条路径的随机数实现不同。历史效果数字来自已有归档，本次未重新运行全队列或重新挑选参数。部署证明的观察边界和时序限制见[机制说明](README.md)。

审核时发现原研究复现入口 original_study.py 在未设置环境变量时会进入第一轮消融，虽然 load_original() 已正确加载第二轮定版配置。已统一复现入口的缺省设置为第二轮，并核对该设置生效；第一轮仍可用显式环境变量重放。

[原始配置归档](../outputs/m7/physionet_fusion_20261120/frozen.json) · [原始逐预算汇总](../outputs/m7/physionet_fusion_20261120/summary.csv) · [原 M8 结果](original_results.md) · [补测结果](RESULTS.md)

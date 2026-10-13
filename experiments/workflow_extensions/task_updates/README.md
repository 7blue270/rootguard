# task_updates：多属性变化与任务变化的探索

本轮独立目录，不修改 root_updates 或 target_mismatch。只读复用它们经过验证的数值组件，prior_sources.json 记录前轮文件哈希，报告脚本检查不变。包含5项机制测试和2项事件指标正/负对照。

目的：量化抵消/放大，并对照验证集调优周期、公共计划刷新和简化有限重启 AboveThreshold。连续任务值为主，不刻意构造风险阈值事件；原风险 RCE 为辅助。不是新机制开发或完整 LLM 实验。

数据：NHANES 200 人/模板，前60验证/后140测试；合成 H=8/16。真实 MIMIC-IV Demo2.2 CBC H=8，20验证/33测试；数据路径复用 root_updates/data.py 的只读路径。均值沿用原 holdout。公开域内按统一幅度缩小代替裁剪，保持精确抵消；时刻按患者随机分散。

隐私：数值轨迹 d_inf 距离隐私 B，时间/结构/计划公共。标准轨迹 E=999B 为充分校准，单独跑 E=4。AboveThreshold 使用全局敏感度、阈值 Lap(2Δ/eps)、查询 Lap(4Δ/eps)，每块至多一个阳性、最多C块，每块 B*f/C；每次发布 B*(1-f)/(C+1)。Oracle 不保护触发信息，仅诊断。验证集当作公开调参资料，保护声明针对测试用户发布接口。

复现（仓库根目录；Python 需 NumPy/SciPy）：

```bash
python -m unittest discover -s experiments/workflow_extensions/task_updates/tests -v
python experiments/workflow_extensions/task_updates/run_experiments.py --config experiments/workflow_extensions/task_updates/configs/exploration.json --out experiments/workflow_extensions/outputs/task_updates/exploration_20261013
python experiments/workflow_extensions/task_updates/period_sweep.py experiments/workflow_extensions/outputs/task_updates/exploration_20261013
python experiments/workflow_extensions/task_updates/oracle_probe.py experiments/workflow_extensions/outputs/task_updates/exploration_20261013
python experiments/workflow_extensions/task_updates/analyze_report.py experiments/workflow_extensions/outputs/task_updates/exploration_20261013
```

输出目录需不存在；修改运行标识以重复实验。完整报告 REPORT.md 与所有统计产物在 outputs/task_updates 下，沿用仓库忽略规则。代码/配置可提交，未自动提交。

参考：[Dwork–Roth Algorithm 1/2, §3.6](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)。当前是该基础思路的简化参考实现，不声称完整复现某篇流式应用论文。敏感度证据与组成规则见 release.py、测试和 REPORT.md。

oracle_probe.py 另做8个非DP诊断：保持Oracle触发时刻和初始缓存不变，仅去掉刷新噪声，定位发布噪声的贡献。它不是可部署机制，不能用其账本标签声称DP保护。period_sweep.py 的全部周期结果用于描述权衡，不用测试集重新选参数。

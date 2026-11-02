# 阶段0：风险类别事件审计

只统计未加噪真值事件、核验论文RCE和普通CMR公式、输出启动门槛判断。不调用A/B、SVT、Oracle或加噪，不修改旧实验结果。门槛为同真实队列至少50事件、20有事件独立患者。

```bash
python -m unittest discover -s experiments/workflow_extensions/task_updates/stage0/tests -v
python experiments/workflow_extensions/task_updates/stage0/audit.py --out experiments/workflow_extensions/outputs/task_updates/stage0_20261013
```

从仓库根运行，需NumPy/SciPy及原MIMIC-IV Demo2.2文件。输出目录须不存在；重复审计用新运行标识。真实H8队列按上一轮metadata中的患者清单重建，另审计H8全资源和H16全资源；合成轨迹仅检查，不计入真实启动门槛。

当轮顺序：先测量、再判断刷新、最后当轮回答。类别事件从第二轮开始，响应当轮延迟为0。A/B不在此阶段执行；四组Oracle诊断和公平性约束仅在报告中记录。

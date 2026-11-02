# Stage1 A/B：研究价值判定

协议详见PROTOCOL.md，固定B诊断案例见configs/formal.json。复用上一轮数值组件，保留旧源码哈希；新增实现位于本目录。

```bash
python -m unittest discover -s experiments/workflow_extensions/task_updates/stage1_testab/tests -v
python experiments/workflow_extensions/task_updates/stage1_testab/run.py --config experiments/workflow_extensions/task_updates/stage1_testab/configs/precheck.json --out experiments/workflow_extensions/outputs/task_updates/stage1_precheck_v2_20261013
python experiments/workflow_extensions/task_updates/stage1_testab/run.py --config experiments/workflow_extensions/task_updates/stage1_testab/configs/formal.json --out experiments/workflow_extensions/outputs/task_updates/stage1_formal_v2_20261013
python experiments/workflow_extensions/task_updates/stage1_testab/analyze.py experiments/workflow_extensions/outputs/task_updates/stage1_formal_v2_20261013
```

需NumPy/SciPy与既有MIMIC-IV Demo2.2本地文件；本次Python为/root/miniconda3/envs/rootguard/bin/python。在仓库根运行，输出目录必须不存在。运行保存配置、来源SHA、完整本目录源码快照、患者清单、验证结果/选参、测试患者统计及报告。前轮文件只读，不自动commit/push。

初次预检查完成核心流程后补充预算份额和冻结日程诊断，再执行v2预检查；正式结果以v2通过后的代码为准。9项测试涵盖状态错误、相邻事件、重复报警、RCE/CMR、压力规则、固定次数、噪声组合和检测预算等。

最终正式目录为stage1_formal_v2：报告前复核修正“合法迟到响应不应算稳定期误报”，补充用例后完整重跑验证选参与测试。原stage1_formal输出保留，不作为最终证据。运行时源文件与分析脚本的SHA分别记录，完整代码快照保留。

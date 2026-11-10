# RootGuard动态根更新数据集

> **格式说明（2026-11-10）**：根目录四份 JSON 是精简对话版，仅保留对话字段和公式；下文部分元数据说明对应原构建器的完整记录。旧 `dataset_validation.json` 的哈希对应完整版本，不能当作当前精简文件的校验。数值可通过当前校验器回放；来源和时间元数据可从本地原始数据恢复。
>
> 新有限缓存实验使用独立的 `data_mine/finite_cache/*.jsonl`，保留同住院批次、实际被测集合（含未变值）、结果可用时间与来源，同时补建 MELD_NA。生成数据已被 Git 忽略，原精简 JSON 不覆盖。新建窗口按最早合格住院和最早时间选择，不按任务类别事件筛选。

新版入口（仓库根，Python 3.12）：

```bash
python -B -m data_mine.build_datasets --format finite
python -B -m data_mine.validate_datasets --format finite --input data_mine/finite_cache \
  --report data_mine/finite_cache/validation.json
python -B -m data_mine.validate_datasets --format compact --input data_mine
```

`build_datasets` 默认导出新格式；完整旧格式仅在显式 `--format legacy --output <新空目录>` 时导出。校验器默认只读，只有显式 `--report` 才写报告。原始来源默认位于 `/home/rootguard_research_audit/datasets/mimic-iv-demo-2.2`，可用 `--source` 指定。详细协议见 `experiments/workflow_extensions/finite_cache/README.md`。

来源：MIMIC-IV Clinical Database Demo 2.2真实检验值。对话文本、查询顺序和轮次安排为确定性合成；不是自然采集的医患对话。未调用LLM或运行DP机制。

|文件|患者数|每人问答轮数|每人更新数|有类别变化的患者|
|---|---:|---:|---|---:|
|ANEMIA.json|20|40|16–24|18|
|FIB4.json|20|40|16–24|17|
|NLR.json|2|40|16–17|2|
|TYG.json|6|40|16–21|1|

合计48条患者—模板记录、1,920轮；跨模板去重为25位患者。同一患者可出现在不同模板，不能按48位独立患者统计。

## 构建与筛选

- 每文件顶层是模板名，值为记录列表，兼容原RootGuard的基本结构。保留id、topic、target_attribute、target_risk、turns、private_value及派生公式。
- 每模板目标20人；当前真实来源仅支持NLR 2人、TYG 6人，不复制患者或伪造测量补齐。
- 每人仅采用一次住院中的轨迹。至少16次真实数值更新；期望更新数按患者排序循环设为16–24次，受实际容量约束。初始化不计更新。
- 初始化从完整同时刻属性组建立；后续允许部分根更新。相同charttime的同批结果合并为一次原子更新，并等该批相关结果全部可用后引入。拒绝冲突重复值；相同数值的重复测量不算数值更新。
- 状态语义是“最新已知测量值”，各根的测量时间可能不同，不能冒充同一时刻完整临床评分。root_measurement_times、root_source_ids及measurements保留来源。
- 同一患者只选一次住院与一个连续更新窗口。固定规则优先覆盖类别变化、无关根变化及不同影响类型，再以较早窗口打破平局；不使用任何DP方法结果选数据。属于有目的筛选，不能用于估计临床事件率。
- 轨迹窗口可能继承窗口之前已经可用的根值，其原始检验行ID仍完整保留；未插值、未修改数值、未混入其他患者。
- FIB4年龄根据anchor_age与锚定年份估计，在该住院内固定；age_provenance记录参照时刻。
- NLR仅使用绝对计数；TYG采用实际记录的血糖，未核实空腹条件，因此只是模板数值任务，不是已验证的临床诊断。

## 更新与答案标注

- root_update轮的private_value仅含改变的根。root_state为更新后的完整最新已知根状态。
- reference_derived_state、reference_parent_versions和expected_task_output是评估真值，不是额外对话发布，也不能作为代理可见的输入。受根更新影响的后代会生成新参考版本，未受影响的版本保持不变。
- 非更新轮询问派生值或分类，回答使用最新根重新计算的参考值；不是加噪发布结果。
- 更新轮占40%–60%；不同患者不强求同一比例。40轮不等于40次检查。
- class_change表示分类改变；类别不变时按目标绝对相对变化<=20%和>20%划分small/large，仅为描述标签，不是临床标准或DP触发阈值。
- ANEMIA的仅RBC更新标记task_irrelevant_root_update；它影响MCV/MCH但不影响MCHC。其余三个模板未人为添加无关根。
- ANEMIA的任务是MCHC类别，类别编号大小不代表临床风险的严格排序。target_risk是最后状态，initial_target_risk是初始状态。
- 当前split为unassigned，尚未建立验证/测试划分；后续必须按跨模板患者ID整体划分。
- TYG仅1位患者出现类别变化，其他无事件轨迹保留，不能将其任务事件召回率记为0。

## 校验与复现

从仓库根运行：

```bash
python -B data_mine/build_datasets.py
python -B data_mine/validate_datasets.py
```

独立校验通过：1,920轮、5,440个根来源引用。核对原始患者/住院/测量值/可用时间，按原RootGuard静态公式和分类函数重算，检查版本、更新比例、文本与数值一致性。详见dataset_validation.json。

dataset_manifest.json记录实际规模、缺口、事件类型、来源哈希及构建脚本哈希。sources保留年龄原始表，feasibility保留容量审计。

此前的三份审核样例JSON及配套REVIEW.md、provenance.json、validation.json已移除。旧build_review_samples.py仅作为既有容量审计的代码依赖保留，不属于本轮数据集。

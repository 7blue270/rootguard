# M6 finite-session diagnostic

Implements `/home/M6_机制设计与优化逻辑.md`; concrete constants and the
terminal-hold risk approximation are documented in `MECHANISM-REVIEW-zh.md`.
Existing M5 and original RootGuard M-Opt observations are reused verbatim.

```bash
cd /home/rootguard-research
/root/miniconda3/envs/rootguard/bin/python -m unittest experiments.workflow_extensions.m6.test_m6 -v
OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m6.run --precheck --output experiments/workflow_extensions/outputs/m6/precheck_reproduce
OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m6.run --output experiments/workflow_extensions/outputs/m6/real_reproduce
/root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m6.diagnostics --output experiments/workflow_extensions/outputs/m6/real_reproduce
```

Output directories must be empty. The formal run uses frozen 54 budget points,
R=20, paired patient and patient-plus-mechanism bootstrap=1000. Primary metrics
use full-trajectory final-node errors; intermediate-query wMAPE, ordinal RCE,
mixed-kernel MAP and release-endpoint attacks are separate. The previously used
test set supports mechanism diagnosis, not fresh generalization confirmation.

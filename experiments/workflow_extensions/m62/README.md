# M6.2: pre-session static/update choice

Main question: can a whole-session public predictor choose between uniform
static S and fixed two-update D, and achieve useful gains over defaults,
patient-trained lookup and validation-tuned periodic release?

```bash
cd /home/rootguard-research
/root/miniconda3/envs/rootguard/bin/python -m unittest experiments.workflow_extensions.m62.test_m62 -v
OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m62.run --precheck --output experiments/workflow_extensions/outputs/m62/precheck_reproduce
OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/rootguard/bin/python -m experiments.workflow_extensions.m62.run --output experiments/workflow_extensions/outputs/m62/real_reproduce
```

Output directories must be empty. Formal run: six budgets, 12000 public paths
per configuration, R20 actual releases, patient LOPO, 1000 paired and nested
bootstrap draws with global patient binding. Public sensitivity is never used
to select the best test prior or drift. The existing final-node query turns
define the endpoint. Training and evaluation release noise are separate.

See `MECHANISM-REVIEW-zh.md` for public calendar-design mixture, reference-prior
assumptions, deduplicated L2 estimation and cross-fitted oracle conventions.
No clinically meaningful eta is invented. This previously examined cohort is
an exploratory replication, not independent patient/formula validation.

Outputs include the three main CSV tables, deployment-cost table, raw metrics
and actions, public/fold predictions and freeze hashes, full sensitivity,
bootstrap indices, audit, per-template PDF/SVG/PNG figures and `REPORT-zh.md`.
Figure rendering uses matplotlib and the installed nature-figure QA tools;
PDF collision inspection requires PyMuPDF (the local existing installation is
reused when available). Core release/evaluation does not depend on PDF tools.

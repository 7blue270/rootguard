# Install

> **⚠️ Python 3.10, 3.11, or 3.12 only — not 3.13.**
> `sentencepiece`, `torch`, and a couple of other transitives don't yet
> ship Python 3.13 wheels, so `pyproject.toml` declares
> `requires-python = ">=3.10,<3.13"`. On a 3.13 interpreter `pip install -e .`
> aborts at the resolver step with:
>
> ```
> ERROR: Package 'rootguard' requires a different Python: 3.13.x not in '<3.13,>=3.10'
> ```
>
> If you hit that, recreate the venv with an older interpreter
> (e.g. `python3.12 -m venv .venv` or `/usr/bin/python3.12 -m venv .venv`)
> and rerun the steps below. Tested on Linux with Python 3.11 and 3.12.

## Option A — pip (any virtualenv tool)

```bash
python3.12 -m venv .venv          # NB: not python3.13
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

`pyproject.toml` pins all runtime dependencies. They split into three
groups:

* **Numerical / analysis (RQ1–RQ3 + plotting)** — numpy, scipy, matplotlib,
  networkx, diffprivlib.
* **RQ4 — LLM agent eval** — openai, python-dotenv.
* **`preempt/` sanitizer (NER, FF3 cipher, name lists)** — torch,
  transformers, sentencepiece, names, names-dataset, pyfpe, tqdm,
  protobuf.

Total install size with the torch / transformers wheels is ~1.5 GB.
RQ1–RQ3 do not actually load the NER models, but `preempt/sanitizer.py`
unconditionally imports `preempt/ner.py` at module-load time, which
imports `torch` and `transformers`. So users who only intend
to reproduce RQ1–RQ3 will see the heavy stack get installed; it is not
an option to leave torch/transformers out of the venv. CPU-only torch
wheels are sufficient — no GPU is required.

## Option B — uv (faster, recommended if you have it)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Verifying the install

From the project root:

```bash
python -c "import numpy, scipy, matplotlib, openai; print('ok')"
```

RQ4 ships a small unit-test suite (the other RQs are pure numerical
scripts and rely on the import smoke check above):

```bash
cd rq4_agent_eval
for t in tests/test_smoke.py tests/test_proxy.py tests/test_attack.py tests/test_agent.py; do
  python "$t"
done
```

Expected: each script ends with `All N tests passed`.

## Environment variables (RQ4 only)

RQ4 requires an OpenAI API key. Copy the template and fill in your key:

```bash
cp .env.example .env
# edit .env
```

The runner in `rq4_agent_eval/experiments/run_sweep.py` reads
`OPENAI_API_KEY` from the process environment (load with
`set -a; source .env; set +a` or via `python-dotenv` if you import it).

RQ1, RQ2, RQ3 do **not** need any API access — they are pure-CPU
numerical / mechanism simulations.

## What is *not* shipped

- The 28,800 raw per-session JSON files for RQ4 (~150 MB). Users
  regenerate them by running `experiments/run_sweep.py`. Frozen
  `aggregates.json`, `tables/`, `plots/`, and `SUMMARY.md` are shipped so
  outputs can be verified without an API key.

## Troubleshooting

- **`ERROR: Package 'rootguard' requires a different Python: 3.13.x not in '<3.13,>=3.10'`**: you are on Python 3.13. Recreate the venv with 3.10/3.11/3.12 (see the warning at the top of this file).
- **`scipy.optimize` complains about constraints in `optimal_budget_allocation_*`**: ensure scipy ≥ 1.10. The pyproject pins this.
- **RQ4 hits OpenAI rate limits**: lower `--workers`. We tested 20-way
  concurrency on a Tier-4 account; lower tiers may need 5–10.
- **Bootstrap SE columns are empty in regenerated tables**: this happens
  if a cell had < 30 successful samples. Increase `--n-patients` or check
  the cell-level `parse_failures` counter in `aggregates.json`.

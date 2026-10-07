# Interpretability

Code for Section 5.2 and the interpretability appendices: linear probes and steering, the Jacobian
Lens, and Activation Oracles, all run on the model organism built in `../model-development/`.

Every method is fitted or built on base Qwen3.5-27B and then applied to the organism. Each rollout is
labelled by whether the organism emits the marker (§) in its visible response. Methods are scored by
AUC at 11 extraction points (Appendix H).

## Layout

```
src/uea_interp/          importable library
  common/                anchors and sites, row registry, config, model loading, inspect logs, AUC helpers
  probes/                contrastive directions, steering hook
  jlens/                 lens readout, token buckets
experiments/             one folder per paper result; scripts run from this directory
  00_setup/              download the lens and base model, preflight check
  01_probes/             ten probes, per-site layer choice, marker AUC
  02_steering/           window and strength sweeps
  03_jlens/              anchor readouts, 22-readout grid
  04_oracles/            Activation Oracle training and queries (overlay on upstream)
  05_resampling/         8-sample robustness
cluster/                 HTCondor wrappers and job lists, same numbering as experiments/
configs/                 experiment.yaml, experiments.yaml (row registry), datasets.yaml
data/                    probe training spans, the CSVs behind the paper figures
figures/                 figure and table scripts
```

## Paper result → code

| Result | Run | Output |
|---|---|---|
| Table 18, 21: contrastive directions | `01_probes/fit_direction.py`, `fit_sysprompt_direction.py` (E/NE) | `results/directions/` |
| Table 21: trained probes (EI/TQ/EA × MM/SM) | `build_probe_dataset.py` → `collect_probe_acts.py` → `fit_probes.py`, `fast_massmean.py` | `results/probes/` |
| App. D.1: layer choice per site | `select_direction_layers.py`, `collect_site_acts.py --judge-corpus` | |
| Table 3, Fig. 2 | `collect_site_acts.py` (per cell) → `score_site_probes.py` → `figures/make_probe_x100.py --neutral`, `make_probe_x100_two_sites.py` | `results/site_probe_eval/` |
| Table 19: window selection | `02_steering/`, jobs `steer2_jobs_s1*.txt`, `evretry*.txt` → `score_steer2.py` → `summarise_s1.py` | `results/steer2_s1_summary.txt` |
| Table 20, Table 2: strength sweep | jobs `steer2_jobs_s2_*.txt`, `s2hi.txt` → `score_steer2.py` → `summarise_s2.py` | `results/steer2_s2_summary.txt` |
| Tables 4, 22–24, Fig. 3 | `03_jlens/run_anchor_lens.py` per cell (`cluster/03_jlens/resplast.sub`, run `ws02`) → `compact_cells.py` → `score_grid.py` → `figures/make_jlens_*.py` | `results/ws02/` |
| Table 5, Fig. 4 | `04_oracles/`: `run_cell.py` per (dataset, condition), stages `collect`, `score`, `summarize` → `export.py` → `figures/make_oracle_x100.py` | `oracle_cells/*/summary.json` |
| Table 25 | `05_resampling/build_samp200*.py`, `build_x100.py` → `run_generate.py` ×8 seeds → `replicate_report.py` | `results/sample_robustness*.json` |

## Setup

```bash
cd interpretability
uv sync                                   # or: pip install -e .[figures]
export UEA_INTERP_ROOT=$PWD
export UEA_MODEL_DATA=/path/to/model/outputs    # the model-development/ pipeline's experiments/, runs/, data/, shared/
export UEA_ORGANISM_PATH=/path/to/merged/organism
export UEA_SCRATCH=/path/to/large/scratch       # activations, tens of GB
export HF_HOME=$UEA_SCRATCH/huggingface

python experiments/00_setup/download.py --what lens   # pre-fitted Qwen3.5-27B lens (neuronpedia/jacobian-lens)
python experiments/00_setup/download.py --what base
python experiments/00_setup/verify_setup.py
```

Every script is run from `interpretability/` and writes to `results/`.

### Inputs from `model-development/`

This code does not generate the organism's benchmark rollouts. It reads them, with their marker labels
and inspect logs, from `$UEA_MODEL_DATA/experiments/`. `configs/experiments.yaml` lists the cells
(benchmark × system-prompt condition) that are used. The trained probes use the evaluation-awareness
judge's labels on base-model traces. Their derived span set is shipped as
`data/probe_spans/spans.jsonl`, so `build_probe_dataset.py` only needs rerunning to rebuild it.
`collect_probe_acts.py` still reads the full reasoning traces from the judge run under
`$UEA_MODEL_DATA/experiments/training_benchmark_baseline/`.

### Activation Oracles

The oracle is a LoRA on base Qwen3.5-27B, trained with the upstream recipe of Karvonen et al. It
runs from a checkout of [activation_oracles](https://github.com/adamkarvonen/activation_oracles)
(MIT) at a pinned commit, with our patch and files laid over it:

```bash
bash experiments/04_oracles/setup.sh
export AO_ROOT=$PWD/experiments/04_oracles/activation_oracles
cd $AO_ROOT
torchrun cluster/train_entry.py            # or condor_submit cluster/train_qwen35.sub
python experiments/run_cell.py --dataset bbh --condition x100 --stage collect   # then score, summarize
python export.py
```

Activations are read at layers 16/32/48 (25/50/75% depth) and injected at layer 1. The adapter-off
control reuses the same pipeline.

## Cluster

`cluster/NN_*/` has the HTCondor submit files and wrappers that produced the paper numbers, plus the
job lists (`jobs/`) that set each sweep. Submit from `interpretability/`, for example
`condor_submit_bid 40 cluster/03_jlens/resplast.sub`. GPU jobs expect an 80 GB card.

# PertaBind reproducibility implementation

This repository is an independent, runnable reconstruction of the method described in
*Exploiting mutation-driven pocket remodeling for therapeutic discovery in
osimertinib-resistant non-small cell lung cancer*.


## What is implemented

- PDBbind/CleanSplit-style and MdrDB-style manifest ingestion.
- RDKit SMILES normalization and ETKDG/MMFF or UFF free-conformer generation.
- PDB/SDF parsing, ligand-centred 0-4, 4-6 and 6-8 Angstrom pocket shells.
- Apo sequence and residue-geometry encoders.
- Ligand 2D and free-state 3D encoders.
- Holo protein, bound-ligand and heterogeneous all-atom pocket encoders.
- Weak apo/holo representation alignment and protein/ligand state shifts.
- Shell-adaptive perturbation scoring, cross-shell propagation and residue attribution.
- Holo teacher, apo-compatible student, prediction and representation distillation.
- Two-stage optimization, five-fold grouped CV, early stopping and resumable checkpoints.
- Affinity and direct delta-delta-G regression; resistance classification at 1.36 kcal/mol.
- CASF/MdrDB metrics, paired bootstrap comparison, ensemble inference and virtual screening.

## Repository map

| File | Purpose |
|---|---|
| `processdata.py` | Validate a CSV manifest and build cached graph tensors. |
| `prepare_pdbbind.py` | Parse a licensed PDBbind index and construct a manifest. |
| `predict_apo.py` | Generate ligand-independent ESMFold structures. |
| `dataset.py` | Dataset, graph batching and device transfer. |
| `model.py` | Complete teacher-student PertaBind network. |
| `loss.py` | Affinity, ranking, alignment and distillation objective. |
| `train.py` | Stage-I/Stage-II training and fold checkpointing. |
| `split.py` | Leakage-aware grouped five-fold split generation. |
| `metric.py` | Regression/classification metrics and paired bootstrap. |
| `metic.py` | Compatibility shim for the misspelled filename requested by some workflows. |
| `evaluate.py` | Single-checkpoint or five-model ensemble evaluation. |
| `predict.py` | Prediction for processed samples. |
| `screen.py` | Score a SMILES library with the distilled student. |
| `utils.py` | Configuration, reproducibility, logging and checkpoint helpers. |
| `configs/default.yaml` | Full reconstruction settings and provenance notes. |
| `configs/quickstart.yaml` | Small settings for a pipeline smoke run. |
| `example/manifest.example.csv` | Input schema example (paths are placeholders). |
| `tests/smoke_test.py` | Synthetic forward/loss/backward test. |

## Installation

Python 3.10 or 3.11 and a CUDA-enabled PyTorch build are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

No PyTorch-Geometric or `torch-scatter` installation is required. Message passing and
segment reductions are implemented with core PyTorch, which makes the environment easier
to reproduce.

## Data contract

Create a CSV with one row per protein-ligand observation. Required columns:

```text
sample_id,sequence,smiles,label,apo_pdb,holo_pdb,bound_ligand
```

- `sequence`: one-letter protein sequence. If blank, it is reconstructed from `apo_pdb`.
- `label`: pK for affinity training, or experimental delta-delta-G in kcal/mol for MdrDB.
- `apo_pdb`: ligand-independent predicted structure (the paper uses ESMFold).
- `holo_pdb`: experimental holo protein coordinates. Optional for student-only prediction.
- `bound_ligand`: SDF/MOL/MOL2/PDB coordinates of the crystallographic ligand. Optional
  for student-only prediction.

Recommended optional columns:

```text
task,split,fold,protein_group,ligand_group,mutation,source
```

`task` is `affinity` or `ddg`. `protein_group` and `ligand_group` are used by
`split.py` to reduce leakage. Paths may be absolute or relative to the manifest.

PDBbind is licensed data and is therefore not redistributed. Download PDBbind v2020,
the published CleanSplit membership, CASF-2016, and MdrDB from their official sources,
then construct a manifest in the schema above. The manuscript reports 95,971 processed
MdrDB observations (219 proteins, 2,487 mutations, 432 drugs), but does not publish the
exact row-level cleaning rules; preserve a raw-to-manifest audit trail for numerical
comparison with the paper.

If apo-like structures have not yet been generated:

```bash
pip install -r requirements-esmfold.txt
python predict_apo.py --input sequences.csv --output-dir data/apo \
  --model facebook/esmfold_v1 --revision YOUR_PINNED_REVISION
```

For PDBbind, `prepare_pdbbind.py` converts supported affinity measurements to pK and
joins them to explicit SMILES and optional published split memberships:

```bash
python prepare_pdbbind.py \
  --index-file /path/to/INDEX_general_PL_data.2020 \
  --pdbbind-root /path/to/PDBbind_v2020 \
  --apo-dir data/apo \
  --smiles-csv data/pdbbind_smiles.csv \
  --membership data/cleansplit_membership.csv \
  --output data/pdbbind_manifest.csv
```

## Preprocess

```bash
python processdata.py \
  --manifest data/train.csv \
  --output data/processed \
  --workers 8 \
  --seed 2026
```

The command writes one `.pt` tensor bundle per sample, `index.csv`, `failures.csv` and
`preprocess_metadata.json`. Existing valid bundles are reused unless `--overwrite` is set.

Generate grouped folds after preprocessing:

```bash
python split.py \
  --index data/processed/index.csv \
  --output data/processed/folds.csv \
  --n-splits 5 \
  --group-column protein_group
```

For an official CleanSplit/CASF evaluation, do not regenerate those memberships: put the
published `train`, `val` and `test` labels in the manifest and keep CASF entirely external.

## Train

Train one fold:

```bash
python train.py --config configs/default.yaml --fold 0
```

Train all five folds:

```bash
for FOLD in 0 1 2 3 4; do
  python train.py --config configs/default.yaml --fold "$FOLD"
done
```

The manuscript settings used by default are AdamW (`beta1=0.9`, `beta2=0.999`), Stage-I
learning rate `1e-4` with batch size `8`, Stage-II learning rate `5e-4` with batch size
`32`, and checkpoint selection by minimum validation RMSE. Epoch counts, hidden size,
layer counts, weight decay and loss weights were not reported and are marked as
reconstruction defaults in the YAML.

For the MdrDB task set `data.task: ddg`. The network predicts delta-delta-G directly; it
does not subtract two independently predicted affinities, matching the manuscript.

## Evaluate

```bash
python evaluate.py \
  --config configs/default.yaml \
  --checkpoints runs/fold_0/best.pt runs/fold_1/best.pt \
                runs/fold_2/best.pt runs/fold_3/best.pt \
                runs/fold_4/best.pt \
  --split test \
  --predictions runs/casf_predictions.csv
```

For `ddg`, the report includes RMSE, MAE, PCC, AUPRC, MCC and F1 using the paper's
resistance threshold of 1.36 kcal/mol. For fixed external benchmarks, report the mean and
standard deviation across the five fold-specific models rather than treating ensemble
averages as five independent test sets.

## Student-only prediction and screening

Processed samples can be scored without holo inputs:

```bash
python predict.py --config configs/default.yaml \
  --checkpoint runs/fold_0/best.pt \
  --index data/processed/index.csv \
  --output predictions.csv
```

Screen a SMILES table against one preprocessed target template:

```bash
python screen.py --config configs/default.yaml \
  --checkpoints runs/fold_0/best.pt runs/fold_1/best.pt \
  --target data/processed/samples/EGFR_template.pt \
  --library compounds.csv \
  --output ranked_compounds.csv
```

The library must contain `compound_id,smiles`. `screen.py` generates free conformers and
uses only the distilled student. It never injects a docked or crystallographic ligand into
the prediction path.

## Reproducibility boundaries

The implementation faithfully follows equations 1-19 and the stated training protocol,
but exact reproduction of the paper's reported numbers is not guaranteed without:

1. the authors' inaccessible repository and exact processed manifests;
2. exact ESMFold version/checkpoints and generated apo structures;
3. hidden dimensions, layer counts, all feature encodings, loss weights and epoch schedules;
4. exact CleanSplit membership/version and MdrDB cleaning decisions;
5. random seeds and the five original fold assignments.

These uncertainties are centralized in `configs/default.yaml`. Replace them with author
values if they become available. The output metadata records the effective configuration,
package versions, input hashes and seed so that every local run remains auditable.

## Minimal checks

```bash
python -m compileall .
pytest -q tests/smoke_test.py
```

The smoke test uses synthetic tensors and does not download models or datasets.

# PertaBind Reproduction Notes

## Reproduction status

This package is an independent, end-to-end reconstruction based on the manuscript's
Methods section and Figure 1B. It is not a mirror of the authors' original source code.
The GitHub URL given in the manuscript was not publicly accessible when this package was
prepared, so the original class structure, tensor dimensions, data-cleaning code, and
random seeds could not be verified.

## Settings explicitly reported in the manuscript

- Inputs: protein sequence, an ESMFold-predicted apo-like structure, a holo
  protein-ligand complex, ligand SMILES, an RDKit-generated and energy-minimized
  free-state ligand conformer, and the ligand conformation extracted from the holo
  complex.
- Pocket definition: residues whose minimum distance from any residue atom to any ligand
  heavy atom is no greater than 8 Angstrom.
- Pocket shells: direct-contact shell at 0-4 Angstrom, near-contact shell at
  4-6 Angstrom, and distal structural shell at 6-8 Angstrom.
- Teacher inputs: global holo-protein representation, bound-state ligand, heterogeneous
  all-atom pocket graph, atom-to-residue pooling, and multi-shell perturbation features.
- Student inputs: protein sequence plus apo-like geometric graph, and a ligand 2D graph
  plus its free-state 3D conformer.
- Distillation: an L1 distance between predictions and an L2 distance between latent
  representations, with stop-gradient applied to the teacher.
- Overall objective: `L_aff + lambda_1 L_rank + lambda_2 L_align + lambda_3 L_dist`.
- Optimizer: AdamW with `beta_1=0.9` and `beta_2=0.999`.
- Stage I: learning rate `1e-4`, batch size `8`.
- Stage II: learning rate `5e-4`, batch size `32`.
- Evaluation: five-fold cross-validation with checkpoint selection by the lowest
  validation RMSE.
- MdrDB task: direct prediction of experimental delta-delta-G; the resistance threshold
  is `delta-delta-G > 1.36 kcal/mol`.
- Regression metrics: RMSE, MAE, and PCC.
- Classification metrics: AUPRC, MCC, and F1.

## Settings not reported in the manuscript

The following values are exposed as configurable reconstruction defaults:

- Hidden dimensions, network depths, attention-head count, RBF count, and dropout.
- `lambda_1`, `lambda_2`, `lambda_3`, the teacher supervision weight, weight decay, and
  gradient clipping.
- Epoch counts, early-stopping patience, and the learning-rate scheduler.
- Exact atom and residue feature encodings and the software or parameters used to derive
  DSSP, solvent-accessible surface area, and residue depth.
- The exact SE(3) architecture, neighborhood size, and geometric cutoffs.
- The precise PDBbind CleanSplit release and row-level MdrDB cleaning procedure.
- The ESMFold checkpoint revision, original fold assignments, and all random seeds.

These choices are centralized in `configs/default.yaml`. Each setting is marked either
`[paper]` or `reconstruction default`. If the authors release the missing details, the
configuration or individual modules can be updated without rewriting the training
pipeline.

## Recommended strict reproduction procedure

1. Freeze the raw dataset versions, licenses, file hashes, and CleanSplit/CASF membership.
2. Pin an immutable ESMFold checkpoint revision and generate all apo-like structures.
3. Use `prepare_pdbbind.py` to create the raw manifest, preserving both original and
   converted affinity labels.
4. Run `processdata.py`, inspect `failures.csv`, and never silently discard failed samples.
5. Freeze `folds.csv` and verify that CASF-2016 and its independent subset are excluded
   from fitting, hyperparameter selection, and checkpoint selection.
6. Train five folds independently, running Stage I before Stage II for every fold.
7. Evaluate all five models separately on each fixed external benchmark and report the
   mean and standard deviation of their metrics.
8. Train the MdrDB task separately and report both regression and resistance-classification
   metrics.
9. Archive the effective YAML configuration, split membership, dependency versions,
   checkpoints, and per-sample predictions.

## Interpretation boundary

Prospective screening must use the student branch only. Teacher-side residue and atom
attributions depend on a holo complex and are appropriate for post-training mechanistic
interpretation; they must not be supplied during unseen-ligand screening, because doing so
would introduce information leakage. `screen.py` and `predict.py` therefore enforce
`student_only=True`, while `interpret.py` explicitly invokes the teacher branch.

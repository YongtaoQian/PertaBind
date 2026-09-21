"""Create reproducible five-fold assignments while preserving external test rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


def make_folds(frame: pd.DataFrame, n_splits: int, seed: int,
               group_column: str | None) -> pd.DataFrame:
    output = frame[["sample_id"]].copy()
    output["fold"] = -1
    output["split"] = "development"
    if "split" in frame:
        external = frame["split"].fillna("").isin(["test", "external_test"])
        output.loc[external, "split"] = frame.loc[external, "split"].values
    else:
        external = pd.Series(False, index=frame.index)
    development_indices = np.flatnonzero(~external.to_numpy())
    if len(development_indices) < n_splits:
        raise ValueError("Fewer development samples than folds")
    if group_column and group_column in frame and frame.loc[~external, group_column].notna().all():
        groups = frame.loc[~external, group_column].astype(str).to_numpy()
        splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(development_indices, groups=groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(development_indices)
    for fold, (_, validation_local) in enumerate(iterator):
        validation_global = development_indices[validation_local]
        output.loc[validation_global, "fold"] = fold
    if (output.loc[~external, "fold"] < 0).any():
        raise RuntimeError("Some development samples were not assigned to a fold")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--group-column", default="protein_group")
    args = parser.parse_args()
    frame = pd.read_csv(args.index)
    folds = make_folds(frame, args.n_splits, args.seed, args.group_column)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(destination, index=False)
    print(f"Wrote {len(folds)} assignments to {destination}")


if __name__ == "__main__":
    main()


"""Generate ligand-independent apo-like structures with ESMFold.

This is intentionally separate from the core requirements because ESMFold checkpoints are
large. Install `requirements-esmfold.txt`, accept the model provider's terms, and pin the
exact checkpoint/revision when matching an archived experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with sample_id,sequence")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="facebook/esmfold_v1")
    parser.add_argument("--revision", help="Immutable model revision for strict reproduction")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError as exc:
        raise SystemExit("Install requirements-esmfold.txt before running this script") from exc

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = EsmForProteinFolding.from_pretrained(
        args.model, revision=args.revision, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    model.trunk.set_chunk_size(args.chunk_size)
    frame = pd.read_csv(args.input)
    missing = {"sample_id", "sequence"} - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="ESMFold"):
        destination = output / f"{row.sample_id}.pdb"
        if destination.exists() and not args.overwrite:
            records.append({"sample_id": row.sample_id, "apo_pdb": str(destination), "status": "cached"})
            continue
        sequence = "".join(str(row.sequence).split()).upper()
        inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_values = model(**inputs)
        pdb = model.output_to_pdb(output_values)[0]
        temporary = destination.with_suffix(".pdb.tmp")
        temporary.write_text(pdb, encoding="utf-8")
        temporary.replace(destination)
        records.append({"sample_id": row.sample_id, "apo_pdb": str(destination), "status": "ok"})
    pd.DataFrame(records).to_csv(output / "apo_index.csv", index=False)
    (output / "esmfold_metadata.json").write_text(json.dumps({
        "model": args.model, "revision": args.revision, "chunk_size": args.chunk_size,
        "torch": torch.__version__, "device": str(device),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


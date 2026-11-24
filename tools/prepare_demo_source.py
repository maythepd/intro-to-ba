"""Materialize the exact seven-table Kaggle release used by Module 5."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from api.prepare_module5 import DATASET
from demo_data.generate import SOURCE, load_sources
from segmentation.data import REQUIRED


def prepare(source: Path, output: Path = SOURCE):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty; existing source data will not be overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as folder:
        staging = Path(folder) / "source"
        staging.mkdir()
        for name in (*[f"{name}.csv" for name in REQUIRED], "generation_manifest.json"):
            matches = list(Path(source).rglob(name))
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name} in pinned source release")
            shutil.copy2(matches[0], staging / name)
        _, audit = load_sources(staging)
        if output.exists():
            output.rmdir()  # Verified empty above; no recursive removal.
        staging.replace(output)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing pinned release-3 directory; otherwise download release 3")
    parser.add_argument("--output", type=Path, default=SOURCE)
    args = parser.parse_args()
    source = args.source
    if source is None:
        import kagglehub
        source = Path(kagglehub.dataset_download(DATASET))
    audit = prepare(source, args.output)
    print(f"Validated {len(REQUIRED)} source tables from {DATASET}: {args.output}")
    for name, entry in audit.items():
        print(f"{name}: {entry['sha256']}")


if __name__ == "__main__":
    main()

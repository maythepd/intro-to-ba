"""Pack or install a trusted, checksum-pinned private serving/demo bundle."""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import urllib.request
from zipfile import ZipFile, ZIP_DEFLATED

from monitoring.settings import ROOT

FILES = (
    "data/demo/customer_features.csv", "data/demo/api_examples.jsonl", "data/demo/manifest.json",
    "artifacts/module5/preprocessor.joblib", "artifacts/module5/manifest.json",
    "artifacts/module5/monitoring_reference.json", "artifacts/module5/monitoring_replay.csv",
    "artifacts/segmentation/cluster_bundle.joblib", "artifacts/segmentation/metadata.json",
    "data/official/module5_customer_features.csv", "data/official/customer_features.csv",
)
LABELS = ("data/demo/synthetic_outcomes.csv", "data/demo/synthetic_outcomes.csv.json")
PACKAGES = ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")
MAX_BYTES = 100 * 1024 * 1024


def environment():
    return {"python": ".".join(platform.python_version_tuple()[:2]),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES}}


def verify_serving(root=ROOT):
    # Lazy imports keep packing/downloading free of model initialization and database access.
    from api import module5, segmentation_service
    from demo_data.labels import read_labels
    from demo_data.validate import check_dataset, model_paths, read_outputs, score_batch
    import numpy as np
    import pandas as pd
    model = module5.load()
    cluster = segmentation_service.load(root / "data/official/customer_features.csv")
    if model is None or cluster is None:
        raise ValueError("Both fitted serving models are required")
    manifest, frame, examples = read_outputs(root / "data/demo")
    for name, path in model_paths().items():
        if not path.is_file() or module5.file_hash(path) != manifest["models"][name]["sha256"]:
            raise ValueError(f"Bundled serving artifact mismatch: {name}")
    if model.input_columns != manifest["input_columns"] or manifest["inference"]["status"] != "passed":
        raise ValueError("Demo inputs/inference are incompatible with bundled model")
    ids = pd.read_csv(module5.DATA_PATH, usecols=["customer_id"]).customer_id
    check_dataset(frame, examples, manifest, ids)
    probabilities, clusters = score_batch(frame, manifest["input_columns"], model, cluster)
    if not np.isfinite(probabilities).all() or len(clusters["kmeans"]) != len(frame):
        raise ValueError("Bundled models failed inference")
    path = root / LABELS[0]
    if path.exists():
        read_labels(path, frame, manifest["output_sha256"]["customer_features.csv"])


def pack(path: Path, *, root: Path = ROOT):
    verify_serving(root)
    names = [*FILES, *(LABELS if (root / LABELS[0]).is_file() else ())]
    metadata = {"version": 1, "environment": environment(),
                "files": {name: sha256((root / name).read_bytes()).hexdigest() for name in names}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, name)
        archive.writestr("bundle.json", json.dumps(metadata, sort_keys=True))
    checksum = sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(checksum + "\n", encoding="ascii")
    return checksum


def install(path: Path, expected: str, *, root: Path = ROOT, check_environment=True):
    if len(expected) != 64 or sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Private bundle SHA256 mismatch")
    with ZipFile(path) as archive:
        entries = archive.infolist()
        if len({item.filename for item in entries}) != len(entries) or sum(item.file_size for item in entries) > MAX_BYTES:
            raise ValueError("Duplicate or oversized bundle entries")
        metadata = json.loads(archive.read("bundle.json"))
        names = set(metadata["files"])
        if (metadata["version"] != 1 or not set(FILES).issubset(names) or names - set(FILES + LABELS)
                or names.intersection(LABELS) not in (set(), set(LABELS))
                or {item.filename for item in entries} != names | {"bundle.json"}):
            raise ValueError("Unexpected bundle paths or incomplete files")
        if check_environment and metadata["environment"] != environment():
            raise ValueError("Serving bundle dependency versions differ; install the pinned runtime requirements")
        with TemporaryDirectory(dir=root) as folder:
            staging = Path(folder)
            for name in names:
                content = archive.read(name)
                if sha256(content).hexdigest() != metadata["files"][name]:
                    raise ValueError("Bundled file checksum mismatch")
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            # All files and paths are checked before publishing any file. Build fails before serving on validation failure.
            for name in sorted(names):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                (staging / name).replace(target)
    if check_environment:
        verify_serving(root)


def download(url: str, expected: str):
    if not url.startswith("https://"):
        raise ValueError("Private artifact download requires HTTPS")
    # URL may contain a signed credential. Neither URLs nor HTTP exception bodies are printed.
    try:
        with TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "bundle.zip"
            with urllib.request.urlopen(url, timeout=90) as response, path.open("wb") as handle:
                received = 0
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > MAX_BYTES:
                        raise ValueError("Private bundle exceeds size limit")
                    handle.write(chunk)
            install(path, expected)
    except Exception:
        raise ValueError("Private bundle download/validation failed; credentials omitted") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pack", type=Path)
    group.add_argument("--install", type=Path)
    group.add_argument("--from-env", action="store_true", help="Install DEMO_ARTIFACT_URL with DEMO_ARTIFACT_SHA256")
    parser.add_argument("--sha256")
    args = parser.parse_args()
    if args.pack:
        print("Private bundle SHA256: " + pack(args.pack))
    elif args.install:
        install(args.install, args.sha256 or "")
        print("Private bundle verified and installed")
    else:
        url, checksum = os.getenv("DEMO_ARTIFACT_URL"), os.getenv("DEMO_ARTIFACT_SHA256")
        if not url or not checksum:
            raise SystemExit("Configure private DEMO_ARTIFACT_URL and DEMO_ARTIFACT_SHA256 before building")
        download(url, checksum)
        print("Private bundle verified and installed")


if __name__ == "__main__":
    main()

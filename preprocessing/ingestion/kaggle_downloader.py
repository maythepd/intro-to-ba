"""
Centralized Kaggle Dataset Ingestion & Downloader Utility.

Canonical dataset references:
  - Kaggle Web URL: https://www.kaggle.com/datasets/maythepd/full-dataset-intro2ba
  - Kaggle Dataset Slug: maythepd/full-dataset-intro2ba
  - Kaggle Direct Download API URL: https://www.kaggle.com/api/v1/datasets/download/maythepd/full-dataset-intro2ba

Automatically ensures all 7 required relational tables are present in the target directory:
  1. kaggle_bank_churn.csv
  2. customer_context.csv
  3. accounts.csv
  4. savings_cycles.csv
  5. complaints.csv
  6. login.csv
  7. transactions.csv
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Optional, Sequence, Union
from urllib.request import urlopen
import zipfile

KAGGLE_DATASET_SLUG: str = "maythepd/full-dataset-intro2ba"
KAGGLE_DATASET_URL: str = "https://www.kaggle.com/datasets/maythepd/full-dataset-intro2ba"
KAGGLE_DIRECT_DOWNLOAD_URL: str = "https://www.kaggle.com/api/v1/datasets/download/maythepd/full-dataset-intro2ba"

REQUIRED_TABLES: tuple[str, ...] = (
    "kaggle_bank_churn.csv",
    "customer_context.csv",
    "accounts.csv",
    "savings_cycles.csv",
    "complaints.csv",
    "login.csv",
    "transactions.csv",
)

AUXILIARY_METADATA_FILES: tuple[str, ...] = (
    "generation_manifest.json",
    "synth_data_dict.md",
)


def verify_dataset_files(data_dir: Union[str, Path], required: Sequence[str] = REQUIRED_TABLES) -> bool:
    """Checks whether all required tables exist in data_dir."""
    target = Path(data_dir)
    if not target.exists() or not target.is_dir():
        return False
    return all((target / name).exists() and (target / name).stat().st_size > 0 for name in required)


def download_via_kaggle_cli(target_dir: Path, dataset_slug: str = KAGGLE_DATASET_SLUG, verbose: bool = True) -> bool:
    """Attempts to download and unzip the dataset using Kaggle CLI."""
    if verbose:
        print(f"[INFO] Attempting download via Kaggle CLI: {dataset_slug} -> {target_dir}...")
    try:
        # Test if kaggle executable is available
        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(target_dir),
            "--unzip",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0 and verify_dataset_files(target_dir):
            if verbose:
                print(f"[OK] Successfully downloaded dataset via Kaggle CLI to {target_dir}")
            return True
        if verbose:
            print(f"[WARN] Kaggle CLI download returned code {result.returncode}: {result.stderr.strip() or result.stdout.strip()}")
    except Exception as exc:
        if verbose:
            print(f"[WARN] Kaggle CLI download failed: {exc}")
    return False


def download_via_kagglehub(target_dir: Path, dataset_slug: str = KAGGLE_DATASET_SLUG, verbose: bool = True) -> bool:
    """Attempts to download the dataset using the official kagglehub library."""
    if verbose:
        print(f"[INFO] Attempting download via kagglehub library: {dataset_slug}...")
    try:
        import kagglehub

        downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
        if downloaded_path.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            for file_path in downloaded_path.rglob("*"):
                if file_path.is_file():
                    dest = target_dir / file_path.name
                    if not dest.exists():
                        shutil.copy2(file_path, dest)
            if verify_dataset_files(target_dir):
                if verbose:
                    print(f"[OK] Successfully copied dataset from kagglehub cache to {target_dir}")
                return True
    except Exception as exc:
        if verbose:
            print(f"[WARN] kagglehub download failed: {exc}")
    return False


def download_via_direct_url(
    target_dir: Path,
    url: str = KAGGLE_DIRECT_DOWNLOAD_URL,
    required: Sequence[str] = REQUIRED_TABLES,
    verbose: bool = True,
) -> bool:
    """Fallback: Downloads dataset zip directly via URL and extracts into target_dir."""
    if verbose:
        print(f"[INFO] Attempting direct HTTP archive download from {url}...")
    archive = target_dir.parent / "kaggle_source_dataset.zip"
    temporary = archive.with_suffix(".downloading")
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not archive.exists():
            with urlopen(url, timeout=180) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(archive)

        with zipfile.ZipFile(archive) as source:
            namelist = source.namelist()
            all_expected = list(required) + list(AUXILIARY_METADATA_FILES)
            for name in all_expected:
                # Find matching filename in zip (ignoring directories)
                match = next((n for n in namelist if Path(n).name.lower() == name.lower()), None)
                if match:
                    dest = target_dir / name
                    if not dest.exists():
                        with source.open(match) as incoming, dest.open("wb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)

        # Write download manifest
        hasher = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)

        manifest = {
            "source_url": url,
            "dataset_slug": KAGGLE_DATASET_SLUG,
            "archive_sha256": hasher.hexdigest(),
        }
        (target_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if verify_dataset_files(target_dir, required=required):
            if verbose:
                print(f"[OK] Direct download completed and verified at {target_dir}")
            return True
    except Exception as exc:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
        if verbose:
            print(f"[WARN] Direct URL download failed: {exc}")
    return False


def ensure_dataset_available(
    target_dir: Union[str, Path] = "data/full_dataset",
    force_download: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Ensures that the 7 required relational CSV tables from the canonical Kaggle dataset
    (maythepd/full-dataset-intro2ba) are present and valid in target_dir.

    If missing or force_download is True, automatically retrieves them through:
      1. kaggle CLI (with user credentials)
      2. kagglehub API
      3. direct Kaggle public download URL fallback

    Returns:
      Resolved Path to the data directory containing the 7 tables.
    """
    target = Path(target_dir).resolve()

    if not force_download and verify_dataset_files(target):
        if verbose:
            print(f"[OK] Dataset verified in '{target}'. All 7 required tables are present.")
        return target

    if verbose:
        print(f"[INFO] Dataset files missing or download requested in '{target}'.")
        print(f"       Connecting to Kaggle repository: {KAGGLE_DATASET_URL}")

    target.mkdir(parents=True, exist_ok=True)

    # Strategy 1: Kaggle CLI
    success = download_via_kaggle_cli(target, dataset_slug=KAGGLE_DATASET_SLUG, verbose=verbose)

    # Strategy 2: kagglehub Python client
    if not success:
        success = download_via_kagglehub(target, dataset_slug=KAGGLE_DATASET_SLUG, verbose=verbose)

    # Strategy 3: Direct URL download
    if not success:
        success = download_via_direct_url(target, url=KAGGLE_DIRECT_DOWNLOAD_URL, verbose=verbose)

    if not success or not verify_dataset_files(target):
        missing = [f for f in REQUIRED_TABLES if not (target / f).exists()]
        raise RuntimeError(
            f"Failed to obtain Kaggle dataset '{KAGGLE_DATASET_SLUG}'. Missing tables: {missing}.\n"
            f"Please verify network access to {KAGGLE_DATASET_URL} or configure Kaggle API credentials."
        )

    return target

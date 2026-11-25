"""Download the group's published raw release, without a Kaggle client dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile

URL = "https://www.kaggle.com/api/v1/datasets/download/maythepd/full-dataset-intro2ba"
FILES = ["kaggle_bank_churn.csv", "accounts.csv", "transactions.csv", "login.csv",
         "complaints.csv", "savings_cycles.csv", "customer_context.csv"]


def download(raw_dir="data/full_dataset"):
    target = Path(raw_dir).resolve()
    archive = target.parent / "source_dataset.zip"
    if all((target / name).exists() for name in FILES) and not archive.exists():
        print(f"[SEG] Raw inputs already present: {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with urlopen(URL, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        if not set(FILES).issubset(source.namelist()):
            raise ValueError("Published archive does not contain the expected seven raw tables")
        for name in FILES + ["generation_manifest.json", "synth_data_dict.md"]:
            if name in source.namelist() and not (target / name).exists():
                with source.open(name) as incoming, (target / name).open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
    hasher = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    (target / "source_download.json").write_text(json.dumps({"url": URL, "archive_sha256": hasher.hexdigest()}, indent=2), encoding="utf-8")
    print(f"[SEG] Downloaded official release to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/full_dataset")
    download(parser.parse_args().raw_dir)

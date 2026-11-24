"""Prepare a normal deployment or install its private prevalidated demo bundle."""
import os
import subprocess
import sys


def main():
    url = os.getenv("DEMO_ARTIFACT_URL")
    checksum = os.getenv("DEMO_ARTIFACT_SHA256")
    if os.getenv("DEMO_BUILD_MODE") == "generate":
        if url or checksum or os.getenv("DEMO_DATA_DIR") != "data/demo":
            raise SystemExit("Generate mode requires DEMO_DATA_DIR=data/demo and no artifact URL/checksum")
        # Build-only work. No production database is opened, migrated or replayed here.
        from api.prepare_module5 import DATASET
        from demo_data.generate import SOURCE, OUTPUT, generate
        from demo_data.labels import generate_labels
        from tools.prepare_demo_source import prepare
        import kagglehub
        from pathlib import Path
        if not SOURCE.exists():
            prepare(Path(kagglehub.dataset_download(DATASET)))
        for module, args in (("api.prepare_module5", ["--source", str(SOURCE)]),
                             ("api.official_data", ["--source", str(SOURCE)]),
                             ("api.segmentation_service", [])):
            subprocess.run([sys.executable, "-m", module, *args], check=True)
        manifest = generate(multiplier=10, max_samples=10000, seed=42)
        if manifest["N_demo"] != 10000 or manifest["inference"]["status"] != "passed":
            raise SystemExit("Synthetic build must pass inference for all 10,000 customers")
        env = {**os.environ, "DATABASE_URL": "sqlite:///artifacts/monitoring/build-validation.sqlite3"}
        subprocess.run([sys.executable, "-m", "demo_data.validate"], env=env, check=True)
        generate_labels(OUTPUT / "customer_features.csv", OUTPUT / "synthetic_outcomes.csv", seed=4242)
        print("Validated five synthetic batches and independent outcomes; no monitoring replay at startup")
    elif url or checksum or os.getenv("DEMO_DATA_DIR"):
        if not url or not checksum or os.getenv("DEMO_DATA_DIR") != "data/demo":
            raise SystemExit("Configure a private bundle or explicit DEMO_BUILD_MODE=generate with DEMO_DATA_DIR=data/demo")
        from tools.demo_artifact import download
        download(url, checksum)
        print("Installed verified serving models and synthetic demo snapshot; replay is an explicit admin operation")
    else:
        for module in ("api.prepare_module5", "api.official_data", "api.segmentation_service"):
            args = [sys.executable, "-m", module]
            if module != "api.segmentation_service":
                args.append("--download")
            subprocess.run(args, check=True)


if __name__ == "__main__":
    main()

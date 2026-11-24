from hashlib import sha256
import json
import os
import subprocess
import sys
from zipfile import ZipFile

import pytest

from tools.demo_artifact import FILES, environment, install


def bundle(tmp_path, *, extra=None, bad_hash=False):
    files = {name: "fixture" for name in FILES}
    if extra:
        files[extra] = "untrusted"
    metadata = {"version": 1, "environment": environment(),
                "files": {name: sha256(value.encode()).hexdigest() for name, value in files.items()}}
    if bad_hash:
        metadata["files"][FILES[-1]] = "0" * 64
    path = tmp_path / "bundle.zip"
    with ZipFile(path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
        archive.writestr("bundle.json", json.dumps(metadata))
    return path, sha256(path.read_bytes()).hexdigest()


def test_install_verifies_all_files_before_publishing(tmp_path):
    path, checksum = bundle(tmp_path, bad_hash=True)
    root = tmp_path / "install"
    root.mkdir()
    with pytest.raises(ValueError, match="file checksum"):
        install(path, checksum, root=root, check_environment=False)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("extra", ["../../escaped.txt", "/absolute.txt", "data/demo/unexpected.txt"])
def test_bundle_rejects_unexpected_paths(tmp_path, extra):
    path, checksum = bundle(tmp_path, extra=extra)
    with pytest.raises(ValueError, match="paths"):
        install(path, checksum, root=tmp_path, check_environment=False)


def test_bundle_rejects_wrong_outer_digest(tmp_path):
    path, _ = bundle(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        install(path, "0" * 64, root=tmp_path, check_environment=False)


def test_demo_enabled_import_and_segmentation_path_are_independent(tmp_path):
    env = {**os.environ, "DEMO_DATA_DIR": str(tmp_path), "CUSTOMERS_CSV_PATH": "data/official/module5_customer_features.csv",
           "SEGMENTATION_CUSTOMERS_CSV_PATH": "data/official/customer_features.csv"}
    result = subprocess.run([sys.executable, "-c",
        "import demo_data.validate; import api.main; from api import segmentation_service as s; "
        "assert str(s.DATA_PATH).replace(chr(92), '/') == 'data/official/customer_features.csv'"],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

"""Check raw files against the official Kaggle release and write the data manifest."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src.common import ARTIFACTS, RAW_DIR, write_json
from src.data.load import CLASSES_FILE, EDGES_FILE, FEATURES_FILE

EXPECTED_SHA256 = {
    FEATURES_FILE: "fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0",
    CLASSES_FILE: "93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493",
    EDGES_FILE: "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = {}
    for name, expected in EXPECTED_SHA256.items():
        path = RAW_DIR / name
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"checksum mismatch for {name}: {actual}")
        files[name] = {"sha256": actual, "bytes": path.stat().st_size}
    manifest = {
        "dataset": "Elliptic Bitcoin transaction dataset",
        "source_url": "https://www.kaggle.com/ellipticco/elliptic-data-set",
        "paper": "https://arxiv.org/abs/1908.02591",
        "license": "CC BY-NC-ND 4.0",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    write_json(ARTIFACTS / "data_manifest.json", manifest)
    print("all checksums match; wrote artifacts/data_manifest.json")


if __name__ == "__main__":
    main()

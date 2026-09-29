#!/usr/bin/env python3
"""Download only the primary Matrix Market file from each SuiteSparse archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
p.add_argument("--manifest", type=Path, default=Path(__file__).with_name("matrices.json"))
a = p.parse_args()
a.directory.mkdir(parents=True, exist_ok=True)
records = []
for case in json.loads(a.manifest.read_text()):
    group, name = case["name"].split("/")
    stem = group + "__" + name
    archive = a.directory / (stem + ".tar.gz")
    url = f"https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        request = urllib.request.Request(url, headers={"User-Agent": "ndgpu-research-benchmark"})
        try:
            with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if case.get("sha256") and digest != case["sha256"]:
        raise RuntimeError(f"checksum mismatch: {archive}")
    matrix = a.directory / (stem + ".mtx")
    with tarfile.open(archive, "r:gz") as source:
        member = source.getmember(f"{name}/{name}.mtx")
        if not member.isfile():
            raise RuntimeError(f"not a regular matrix file: {member.name}")
        with source.extractfile(member) as data, matrix.open("wb") as output:
            shutil.copyfileobj(data, output)
    records.append({**case, "url": url, "sha256": digest, "file": matrix.name})
    (a.directory / "downloads.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"{case['name']}: {archive.stat().st_size:,} bytes, sha256={digest}", flush=True)

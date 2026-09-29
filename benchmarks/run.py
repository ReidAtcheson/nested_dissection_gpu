#!/usr/bin/env python3
"""Run each method separately, sequentially, and keep failures in the record."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("binary", type=Path)
p.add_argument("matrices", type=Path)
p.add_argument("results", type=Path)
p.add_argument("--reps", type=int, default=3)
p.add_argument("--timeout", type=int, default=600)
p.add_argument("--case", action="append")
a = p.parse_args()
a.results.mkdir(parents=True, exist_ok=True)
cases = [{"name": f"grid{s}^3", "n": s**3, "input": f"grid:{s}"} for s in (32, 64)]
for case in json.loads(Path(__file__).with_name("matrices.json").read_text()):
    cases.append({**case, "input": str((a.matrices / (case["name"].replace("/", "__") + ".mtx")).resolve())})
if a.case:
    known = {c["name"] for c in cases}
    if set(a.case) - known:
        p.error("unknown case")
    cases = [c for c in cases if c["name"] in a.case]
records = []
for case in cases:
    prefix = a.results / case["name"].replace("/", "__")
    for method in ("bfs", "cudss"):
        command = [str(a.binary.resolve()), case["input"], case["name"], str(a.reps), str(prefix.resolve()), method]
        print(f"Running {case['name']} / {method}", flush=True)
        with Path(str(prefix) + f".{method}.csv").open("w") as out, Path(str(prefix) + f".{method}.log").open("w") as err:
            try:
                run = subprocess.run(command, stdout=out, stderr=err, timeout=a.timeout)
                status = "ok" if run.returncode == 0 else f"exit {run.returncode}"
            except subprocess.TimeoutExpired:
                status = f"timeout ({a.timeout}s)"
        records.append({"name": case["name"], "n": case["n"], "method": method, "status": status, "command": command})
        (a.results / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"  {status}", flush=True)

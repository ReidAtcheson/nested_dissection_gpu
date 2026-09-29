#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from statistics import median

p = argparse.ArgumentParser()
p.add_argument("results", type=Path)
a = p.parse_args()
records = json.loads((a.results / "runs.json").read_text())
summary = {}
for record in records:
    name, method = record["name"], record["method"]
    row = summary.setdefault(name, {"matrix": name, "n": record["n"]})
    row[method + "_status"] = record["status"]
    if record["status"] != "ok":
        continue
    path = a.results / (name.replace("/", "__") + f".{method}.csv")
    samples = list(csv.DictReader(path.open()))
    assert samples and all(int(s["n"]) == record["n"] for s in samples)
    fills = {int(s["exact_nnz_l"]) for s in samples}
    assert len(fills) == 1, f"nonrepeatable fill: {name}/{method}"
    row[method + "_fill"] = fills.pop()
    for metric in ("cpu_ms", "wall_ms"):
        row[method + "_" + metric] = median(float(s[metric]) for s in samples)
    if method == "bfs":
        for metric in ("levels", "leaves", "largest_leaf", "oversized_leaves", "peak_batch"):
            row[metric] = int(samples[-1][metric])
rows = list(summary.values())
(a.results / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("| Matrix | n | BFS nnz(L) | cuDSS ND nnz(L) | BFS CPU ms | cuDSS CPU ms | BFS wall ms | cuDSS wall ms |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for row in rows:
    name = row["matrix"]
    label = f"[{name}](https://sparse.tamu.edu/{name})" if "/" in name else name.replace("grid", "").replace("^3", "³ grid")
    fields = [label, f"{row['n']:,}"]
    for metric in ("fill", "cpu_ms", "wall_ms"):
        for method in ("bfs", "cudss"):
            key = method + "_" + metric
            fields.append((f"{row[key]:,}" if metric == "fill" else f"{row[key]:,.2f}") if key in row else row.get(method + "_status", "pending"))
    print("| " + " | ".join(fields) + " |")

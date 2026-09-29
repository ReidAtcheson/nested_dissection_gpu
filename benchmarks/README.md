# Matrix Market gauntlet

Ten symmetric SuiteSparse Matrix Collection cases plus the 32³ and 64³ grids.
The fixed selection covers structural mechanics, dense FEM blocks, CFD,
unstructured thermal FEM, and sparse finite differences. Matrix names and
archive SHA-256 hashes are in `matrices.json`; the script downloads the original
Matrix Market archives from the collection and extracts only the primary matrix.

Build with CUDA 12+, cuDSS 0.8+, and CXSparse development headers/library:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DNDGPU_WITH_CUDSS=ON -DCUDSS_ROOT=/path/to/cudss -DNDGPU_BUILD_BENCHMARKS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 benchmarks/fetch.py benchmarks/data
python3 benchmarks/run.py build/ndgpu_compare benchmarks/data benchmarks/results
python3 benchmarks/summarize.py benchmarks/results > benchmarks/results/table.md
```

`run.py` executes one method at a time, with one warmup and three timed samples.
It records raw samples, errors/timeouts, permutations, inverse permutations,
and cuDSS level trees. The summary uses medians and retains unsuccessful cases.
The 600-second per-method timeout includes file loading and fill verification;
it is not an ordering-time measurement. `--case Group/name` selects a case.

Both methods receive the same **stored pattern**, including explicit zeros.
Symmetric Matrix Market storage is expanded and duplicates are removed. The
ordering graph omits diagonal entries; the Cholesky pattern includes every
diagonal. General-storage files are accepted only if structurally symmetric.
cuDSS receives a Laplacian-plus-identity matrix with that pattern, explicitly
configured for nested dissection. No numerical factorization is executed.

Fill is exact scalar `nnz(L)` including the diagonal, computed from each
permutation. Elimination-tree reach counts are checked column by column against
CXSparse's symbolic count algorithm; these are not relaxed supernode counts.
The test also compares with explicit elimination on small permuted graphs.

Wall and process CPU clocks bracket `Ordering::compute()` or cuDSS's
`CUDSS_PHASE_REORDERING`, including completion waits. Upload, graph validation,
workspace/context setup, file I/O, permutation downloads, structural validation,
and exact fill counting are outside the timers. These are standalone ordering
times: importing the BFS result into cuDSS is excluded. The library's GPU audits
and scalar control/statistics transfers remain inside its ordering time.
All BFS cases use the defaults: 16 trials, 128-vertex leaf target, 4 refinement
passes, 64 moves/pass, 32-move stall limit, seed 42, 55% balance, 22-level cap.
Oversized leaves and depth are retained in the raw results; no matrix-specific
tuning is performed. CPU waits use blocking synchronization; clocks are unlocked.

The recorded run uses a Tesla T4 on an 8-vCPU Intel Xeon 2.30 GHz VM,
CUDA 12.9, cuDSS 0.8.0, and the Ubuntu SuiteSparse 5.10.1 CXSparse package.
`reference/samples.csv` preserves all 72 measured samples; `summary.json`
also records leaf/depth diagnostics. All 24 method/case combinations completed.
With the fixed policy, `cant` retained 10 oversized leaves (maximum 144), and
`crankseg_2` retained 19 (maximum 409); other cases met the 128-vertex target.
Neither exception was caused by the depth limit. These outcomes are included,
without changing the algorithm or its settings for those matrices.

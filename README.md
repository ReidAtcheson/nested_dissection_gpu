# Non-production code warning

This is a heavily ai-assisted experiment, not production code.

This uses a simple random BFS based nested dissection with refinement. This is like
METIS but without coarsening and uncoarsening, which could possibly improve the results.

The aim was to see if I could achieve decent fill reduction with minimal CPU involvement. Results
can be found at the end of this document. This definitely achieves the goal of minimizing CPU involvement.
Wall time improvements are mixed and appear best in the most regular cases such as stencils. Fill
reduction results are also mixed, with only one test matrix beating the reported fill
of `cuDSS` own nested dissection (which runs on the CPU), many having roughly comparable fill,
but some having significantly more (up to 2X).

I think these results could be a good starting place for improvements such as introduction of efficient coarsening
and uncoarsening+refinemnet steps. I attempted to include spectral bisection candidates as these favor GPU
much more than BFS but there were no cases in which those were ever selected over plain BFS candidates,
so I didn't include that code.




# nested_dissection_gpu

Uncoarsened GPU nested dissection: 16 randomized BFS trials per subgraph,
32 boundary-refined separator candidates, and independent children batched
by size. This is a research implementation; graph indices are always int32.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/ndgpu_grid 32 3
```

Requires CUDA 12+ and CMake 3.24+. CUB comes with CUDA. Set the architecture
for your GPU; 75 is the Tesla T4 used in the experiments.

```cpp
#include <ndgpu/ordering.hpp>

// Device CSR: zero-based, sorted unique columns, full symmetric pattern.
ndgpu::Graph graph{n, nnz, d_row_offsets, d_column_indices};
ndgpu::Ordering ordering(graph); // owns reusable workspace; borrows graph
ndgpu::Order order = ordering.compute();
// order.permutation[new] = old; order.inverse[old] = new.
// Output arrays stay on the GPU and are owned by ordering.
```

The default leaf target is 128. A subgraph becomes a leaf when small enough,
when no acceptable cut is found, or at the depth limit. Check
`ordering.statistics().oversized_leaves`. `compute()` synchronizes its stream
and reads scalar control data; it is not a fully asynchronous/CUDA-graph API.

For cuDSS 0.8+, enable `-DNDGPU_WITH_CUDSS=ON -DCUDSS_ROOT=/path/to/cudss`.
`ndgpu::set_cudss_order(handle, config, data, order)` from `<ndgpu/cudss.hpp>`
sets the device permutation and level partition tree before reordering.
Keep `ordering` alive while cuDSS consumes them. `ndgpu_cudss` is a complete
SPD solve example. Matrix values are needed only by cuDSS, not the ordering.

## Ordering comparison

Tesla T4, 8-vCPU Intel Xeon 2.30 GHz VM; CUDA 12.9, cuDSS 0.8.0. Medians of
three warm runs, same default BFS settings for every matrix. Fill is exact
scalar `nnz(L)`, including the diagonal, for the stored structural pattern.
Times measure ordering only; uploads, setup, fill counting, and cuDSS import
of our order are excluded.

| Matrix | n | BFS nnz(L) | cuDSS ND nnz(L) | BFS CPU ms | cuDSS CPU ms | BFS wall ms | cuDSS wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32³ grid | 32,768 | 5,464,168 | 5,303,174 | 5.31 | 253.72 | 96.94 | 259.22 |
| 64³ grid | 262,144 | 105,404,552 | 106,823,635 | 11.34 | 2,505.66 | 1,327.82 | 2,534.54 |
| [HB/bcsstk17](https://sparse.tamu.edu/HB/bcsstk17) | 10,974 | 1,303,959 | 1,032,463 | 3.70 | 54.06 | 127.56 | 58.40 |
| [Boeing/bcsstk36](https://sparse.tamu.edu/Boeing/bcsstk36) | 23,052 | 4,482,607 | 2,568,284 | 5.57 | 45.33 | 350.43 | 52.11 |
| [Williams/cant](https://sparse.tamu.edu/Williams/cant) | 62,451 | 23,838,799 | 18,379,023 | 5.83 | 1,328.21 | 1,161.62 | 1,358.02 |
| [Rothberg/cfd1](https://sparse.tamu.edu/Rothberg/cfd1) | 70,656 | 35,992,255 | 19,218,744 | 6.75 | 1,018.85 | 623.87 | 1,040.84 |
| [Schmid/thermal1](https://sparse.tamu.edu/Schmid/thermal1) | 82,654 | 3,340,735 | 2,424,592 | 5.26 | 534.84 | 396.41 | 544.47 |
| [GHS_psdef/apache1](https://sparse.tamu.edu/GHS_psdef/apache1) | 80,800 | 11,120,296 | 10,032,800 | 6.71 | 632.64 | 317.22 | 649.83 |
| [Boeing/pwtk](https://sparse.tamu.edu/Boeing/pwtk) | 217,918 | 58,253,213 | 47,114,605 | 11.06 | 381.35 | 3,268.10 | 452.17 |
| [GHS_psdef/hood](https://sparse.tamu.edu/GHS_psdef/hood) | 220,542 | 59,727,743 | 26,217,961 | 12.27 | 302.90 | 3,827.83 | 371.57 |
| [GHS_psdef/crankseg_2](https://sparse.tamu.edu/GHS_psdef/crankseg_2) | 63,838 | 80,349,459 | 40,968,637 | 12.05 | 365.91 | 8,270.14 | 437.29 |
| [GHS_psdef/apache2](https://sparse.tamu.edu/GHS_psdef/apache2) | 715,176 | 144,059,439 | 131,737,455 | 16.94 | 7,069.42 | 4,837.55 | 7,126.45 |

[Reproduce the suite](benchmarks/README.md) · [Raw samples](benchmarks/reference/samples.csv)

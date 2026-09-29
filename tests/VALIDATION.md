Validated on a Tesla T4 with CUDA 12.9 and cuDSS 0.8.0. The CMake build also
passes with CUDA 12.0 (without cuDSS).

- `ndgpu_test`: permutation/inverse and tree-edge validation, deterministic
  replay, nondefault stream, disconnected/irregular graphs, malformed CSR,
  and depth-limit reporting. Memcheck, synccheck and racecheck report no errors.
- `ndgpu_test --reference-grids`: 32³ and 64³ permutations and level trees
  match the original experiment. Fixed fingerprints are embedded in the test.
  Exact reference nnz(L): 5,464,168 and 105,404,552, respectively.
- `compute-sanitizer --tool memcheck ./ndgpu_grid 33 1`: passes, exercising
  both shared-memory and global-memory refinement.
- `ndgpu_cudss 16`: exact permutation/tree round trip and an SPD solve with
  relative solution error and residual below 1e-10.

Grid timing excludes graph upload and workspace construction; one warmup is
followed by three measured calls. `compute()` includes scalar synchronization
and final statistics. Blocked waits are selected by the example, not the library.

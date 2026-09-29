#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace ndgpu {

// Borrowed device CSR: zero based, sorted unique columns, full symmetric
// pattern. Diagonal entries are allowed and ignored. Keep the graph alive and
// unchanged for the lifetime of Ordering. No matrix values are needed.
struct Graph {
    std::int32_t n;
    std::int32_t nnz;
    const std::int32_t *row_offsets;
    const std::int32_t *column_indices;
};

struct Options {
    std::int32_t trials = 16;
    std::int32_t leaf_size = 128;
    std::int32_t refinement_passes = 4;
    std::int32_t moves_per_pass = 64;
    std::int32_t stall_limit = 32;
    std::int32_t max_levels = 22; // root is level 0; at most 22 occupied levels
    std::uint32_t seed = 42;
    float balance = 0.55f; // heavier child / sum of child sizes, excluding separator
    float max_separator_fraction = 0.5f;
};

// Device pointers owned by Ordering, valid until its next compute() or
// destruction.
struct Order {
    std::int32_t n;
    const std::int32_t *permutation; // permutation[new index] = original vertex
    const std::int32_t *inverse;     // inverse[original vertex] = new index
    const std::int32_t *parts;       // cuDSS: deepest level first, left to right
    std::int32_t num_parts;          // (1 << num_levels) - 1; empty nodes have size 0
    std::int32_t num_levels;
};

struct Statistics {
    std::int32_t root_separator = 0;
    std::int32_t leaves = 0;
    std::int32_t largest_leaf = 0;
    std::int32_t oversized_leaves = 0;
    std::int32_t component_splits = 0;
    std::int32_t peak_batch = 0;
    std::int32_t max_depth = 0;
};

// Reusable device workspace. Construction validates CSR on the GPU and
// allocates storage. compute() performs ordering and returns after the given
// stream finishes. Only scalar control/diagnostic data returns to the host;
// CSR, cuts and ordering stay on device. No CUDA context-wide scheduling policy
// is changed by the library. A single instance is not safe for simultaneous
// compute() calls.
class Ordering {
  public:
    explicit Ordering(Graph graph, Options options = {}, cudaStream_t stream = nullptr);
    ~Ordering();
    Ordering(const Ordering &) = delete;
    Ordering &operator=(const Ordering &) = delete;
    Ordering(Ordering &&) = delete;
    Ordering &operator=(Ordering &&) = delete;

    Order compute();
    Statistics statistics() const;

  private:
    struct Workspace;
    Workspace *workspace_;
};

} // namespace ndgpu

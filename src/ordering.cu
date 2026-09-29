#include "ndgpu/ordering.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cub/cub.cuh>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ndgpu {
namespace {
using I = std::int32_t;
using U64 = unsigned long long;
constexpr I NT = 256;
static_assert(sizeof(I) == sizeof(int), "CUDA atomics require 32-bit int");

void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

// Only ownership and allocation; algorithms use plain device pointers.
struct Arena {
    std::vector<void *> allocations;
    void *bytes(std::size_t size) {
        if (!size)
            return nullptr;
        void *p = nullptr;
        check(cudaMalloc(&p, size));
        try {
            allocations.push_back(p);
        } catch (...) {
            cudaFree(p);
            throw;
        }
        return p;
    }
    I *ints(std::size_t count) { return static_cast<I *>(bytes(count * sizeof(I))); }
    U64 *keys(std::size_t count) { return static_cast<U64 *>(bytes(count * sizeof(U64))); }
    ~Arena() {
        for (void *p : allocations)
            cudaFree(p);
    }
};

struct Node {
    I beg = 0, end = 0;
    I origin = 0; // implicit complete-tree ID, used for seeds and cuDSS export only
    I components = 0, split = 0, child = -1;
    I left = 0, right = 0, sep = 0, final_size = 0;
    U64 component_cut = ~U64(0);
};
struct View {
    I n;
    const I *row, *col, *p, *inv, *owner;
    Node *nodes;
};
__device__ unsigned hash32(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}
__device__ I tree_rank(I id, I levels) {
    I depth = 31 - __clz(id + 1);
    return (1 << levels) - (1 << (depth + 1)) + id - ((1 << depth) - 1);
}
__device__ View slice(View g, I part, I stride) {
    I base = g.nodes[part].beg;
    g.n = stride;
    g.nodes += part;
    g.row += base;
    return g;
}
__device__ I vertices(View g) { return g.nodes[0].end - g.nodes[0].beg; }
__device__ I neighbor(View g, I e) { return g.col[e] - g.nodes[0].beg; }

struct Counts {
    I left, right, sep;
};
struct AddCounts {
    __device__ Counts operator()(Counts a, Counts b) const {
        return {a.left + b.left, a.right + b.right, a.sep + b.sep};
    }
};
// Lexicographic scoring preserves the experiment's balance/separator/imbalance
// policy without cubing the vertex count in an integer scalar score.
struct Score {
    I excess, sep, imbalance;
};
__device__ bool less(Score a, Score b) {
    if (a.excess != b.excess)
        return a.excess < b.excess;
    if (a.sep != b.sep)
        return a.sep < b.sep;
    return a.imbalance < b.imbalance;
}
__device__ Score objective(I l, I r, I s, I n, float balance) {
    I excess = max(0, max(l, r) - I(floorf(balance * (l + r))));
    if (l == 0 || r == 0)
        excess += n;
    return {excess, s, abs(l - r)};
}
struct Move {
    Score score;
    I pos;
};
struct MinMove {
    __device__ Move operator()(Move a, Move b) const {
        if (less(a.score, b.score))
            return a;
        if (less(b.score, a.score))
            return b;
        return a.pos < b.pos ? a : b;
    }
};
struct Trial {
    I left, right, sep;
    Score score;
};

// Validate offsets first so malformed CSR can never cause an out-of-bounds read
// in the later column/symmetry check. No host copy of the graph is needed.
__global__ void validate_offsets(Graph g, I *error) {
    I u = blockIdx.x * NT + threadIdx.x;
    if (u > g.n)
        return;
    I begin = g.row_offsets[u];
    if (begin < 0 || begin > g.nnz || (u == 0 && begin != 0) || (u == g.n && begin != g.nnz) ||
        (u < g.n && begin > g.row_offsets[u + 1]))
        atomicExch(error, 1);
}
__global__ void validate_columns(Graph g, I *error) {
    I u = blockIdx.x * NT + threadIdx.x;
    if (u >= g.n)
        return;
    for (I e = g.row_offsets[u]; e < g.row_offsets[u + 1]; ++e) {
        I v = g.column_indices[e];
        if (v < 0 || v >= g.n || (e > g.row_offsets[u] && g.column_indices[e - 1] >= v)) {
            atomicExch(error, 1);
            continue;
        }
        I lo = g.row_offsets[v], hi = g.row_offsets[v + 1];
        while (lo < hi) {
            I mid = lo + (hi - lo) / 2;
            if (g.column_indices[mid] < u)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo == g.row_offsets[v + 1] || g.column_indices[lo] != u)
            atomicExch(error, 1);
    }
}

__global__ void init_components(int n, int *parents, int *sizes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        parents[i] = i;
        sizes[i] = 0;
    }
}
__device__ int atomic_root(int *p, int a) {
    int b = atomicAdd(p + a, 0);
    while (b != a) {
        a = b;
        b = atomicAdd(p + a, 0);
    }
    return a;
}
__global__ void join_components(View g, int *parents) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= g.n || g.owner[pos] < 0)
        return;
    Node t = g.nodes[g.owner[pos]];
    int u = g.p[pos];
    for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
        int v = g.inv[g.col[e]];
        if (v < t.beg || v >= t.end)
            continue;
        if (v >= pos)
            continue;
        int a = pos, b = v;
        while (true) {
            a = atomic_root(parents, a);
            b = atomic_root(parents, b);
            if (a == b)
                break;
            if (a < b) {
                int c = a;
                a = b;
                b = c;
            }
            if (atomicCAS(parents + a, a, b) == a)
                break;
        }
    }
}
__global__ void count_components(View g, const int *parents, int *root, int *sizes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.n || g.owner[i] < 0)
        return;
    int r = i;
    while (parents[r] != r)
        r = parents[r];
    root[i] = r;
    atomicAdd(sizes + r, 1);
    if (r == i)
        atomicAdd(&g.nodes[g.owner[i]].components, 1);
}
__global__ void component_cut(View g, const int *sizes, const int *prefix, int leaf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.n || g.owner[i] < 0 || sizes[i] == 0)
        return;
    Node t = g.nodes[g.owner[i]];
    int n = t.end - t.beg;
    if (t.components <= 1 || n <= leaf)
        return;
    int left = prefix[i] - (t.beg ? prefix[t.beg - 1] : 0);
    if (left <= 0 || left >= n)
        return;
    U64 key = (U64(abs(n - 2 * left)) << 32) | unsigned(i);
    atomicMin(&g.nodes[g.owner[i]].component_cut, key);
}
__global__ void component_labels(View g, const int *root, const int *prefix, int *labels,
                                 int leaf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.n || g.owner[i] < 0)
        return;
    Node &t = g.nodes[g.owner[i]];
    if (t.components <= 1 || t.end - t.beg <= leaf)
        return;
    int boundary = int(t.component_cut & 0xffffffff);
    labels[i] = (root[i] <= boundary) ? 0 : 1;
    if (i == t.beg) {
        t.left = prefix[boundary] - (t.beg ? prefix[t.beg - 1] : 0);
        t.right = t.end - t.beg - t.left;
        t.sep = 0;
        t.split = 1;
    }
}
__global__ void make_bins(Node *nodes, int first, int count, int leaf, int *counts, int *ids,
                          int stride) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count)
        return;
    int id = first + k;
    Node t = nodes[id];
    int n = t.end - t.beg;
    if (n <= leaf || t.components != 1)
        return;
    int b = 32 - __clz(n - 1);
    int pos = atomicAdd(counts + b, 1);
    ids[b * stride + pos] = id;
}

// One block per BFS trial. Stop at the first complete level reaching half the
// vertices; the following distance/hash sort selects an exact half from it.
__global__ void grow(View g, int trials, unsigned seed, int *distances, int stride) {
    g = slice(g, blockIdx.y, stride);
    distances += size_t(blockIdx.y) * g.n * trials;
    seed ^= hash32(unsigned(g.nodes[0].origin));
    int t = blockIdx.x, n = vertices(g);
    int *d = distances + size_t(t) * g.n;
    using Reduce = cub::BlockReduce<int, NT>;
    __shared__ typename Reduce::TempStorage tmp;
    __shared__ int depth, total, covered, frontier;
    if (threadIdx.x == 0) {
        depth = 0;
        total = n;
        covered = 0;
    }
    for (int u = threadIdx.x; u < g.n; u += NT)
        d[u] = INT_MAX;
    __syncthreads();
    if (threadIdx.x == 0)
        d[hash32(seed + unsigned(t) * 104729u) % n] = 0;
    __syncthreads();
    for (;;) {
        int w = 0, f = 0;
        for (int u = threadIdx.x; u < n; u += NT) {
            int du = atomicAdd(d + u, 0);
            if (du <= depth)
                w += 1;
            if (du == depth)
                ++f;
        }
        int all = Reduce(tmp).Sum(w);
        if (threadIdx.x == 0)
            covered = all;
        __syncthreads();
        all = Reduce(tmp).Sum(f);
        if (threadIdx.x == 0)
            frontier = all;
        __syncthreads();
        if (2 * covered >= total)
            break;
        if (!frontier) {
            if (threadIdx.x == 0) {
                unsigned best = ~0u;
                int root = -1;
                for (int u = 0; u < n; ++u)
                    if (d[u] == INT_MAX) {
                        unsigned h = hash32(unsigned(u) ^ seed ^ unsigned(t) * 9277u);
                        if (root < 0 || h < best) {
                            root = u;
                            best = h;
                        }
                    }
                if (root >= 0)
                    d[root] = depth;
            }
            __syncthreads();
            continue;
        }
        for (int u = threadIdx.x; u < n; u += NT)
            if (atomicAdd(d + u, 0) == depth)
                for (int e = g.row[u]; e < g.row[u + 1]; ++e)
                    atomicCAS(d + neighbor(g, e), INT_MAX, depth + 1);
        __syncthreads();
        if (threadIdx.x == 0)
            ++depth;
        __syncthreads();
    }
}
__global__ void bfs_keys(View g, int trials, unsigned seed, const int *distance, U64 *keys,
                         int *values, int *offsets, int stride) {
    g = slice(g, blockIdx.y, stride);
    size_t base = size_t(blockIdx.y) * g.n * trials;
    distance += base;
    keys += base;
    values += base;
    seed ^= hash32(unsigned(g.nodes[0].origin));
    int i = blockIdx.x * NT + threadIdx.x;
    if (i < g.n * trials) {
        int t = i / g.n, u = i % g.n;
        values[i] = u;
        keys[i] = u < vertices(g) ? (U64(unsigned(distance[i])) << 32) |
                                        hash32(unsigned(u) ^ seed ^ unsigned(t) * 26699u)
                                  : ~U64(0);
    }
    if (i < trials || (blockIdx.y == gridDim.y - 1 && i == trials))
        offsets[blockIdx.y * trials + i] = (blockIdx.y * trials + i) * g.n;
}

__global__ void split_half(View g, int trials, const int *order, int *binary, int stride) {
    g = slice(g, blockIdx.y, stride);
    size_t base = size_t(blockIdx.y) * stride * trials;
    int i = blockIdx.x * NT + threadIdx.x;
    if (i < stride * trials) {
        int t = i / stride, rank = i % stride;
        binary[base + size_t(t) * stride + order[base + i]] = rank < vertices(g) / 2 ? 0 : 1;
    }
}
__global__ void covers(View g, int candidates, const int *binary, int *labels, int stride) {
    g = slice(g, blockIdx.y, stride);
    binary += size_t(blockIdx.y) * g.n * (candidates / 2);
    labels += size_t(blockIdx.y) * g.n * candidates;
    int i = blockIdx.x * NT + threadIdx.x;
    if (i >= g.n * candidates)
        return;
    int t = i / g.n, u = i % g.n, side = t % 2;
    int label = binary[size_t(t / 2) * g.n + u];
    bool boundary = false;
    if (u < vertices(g) && label == side)
        for (int e = g.row[u]; e < g.row[u + 1]; ++e)
            boundary |= binary[size_t(t / 2) * g.n + neighbor(g, e)] != label;
    labels[i] = boundary ? 2 : label;
}
// One block per candidate. Threads score the current boundary, CUB selects a
// deterministic winner, and thread 0 applies the move. Moves may worsen the
// score temporarily; only the best prefix is saved across alternating passes.
__global__ void refine(View g, int *labels, unsigned char *work, int *boundary_list,
                       int *boundary_pos, Trial *out, int passes, int moves, int stall,
                       float balance, int stride) {
    g = slice(g, blockIdx.y, stride);
    size_t base = size_t(blockIdx.y) * g.n * gridDim.x;
    labels += base;
    work += base;
    out += blockIdx.y * gridDim.x;
    boundary_list += base;
    boundary_pos += base;
    int t = blockIdx.x, n = vertices(g), total = n;
    int *saved = labels + size_t(t) * g.n;
    int *bnd = boundary_list + size_t(t) * g.n;
    int *bptr = boundary_pos + size_t(t) * g.n;
    extern __shared__ unsigned char shared_labels[];
    unsigned char *current = stride <= 32768 ? shared_labels : work + size_t(t) * g.n;
    using Sum = cub::BlockReduce<Counts, NT>;
    using Min = cub::BlockReduce<Move, NT>;
    __shared__ typename Sum::TempStorage sumtmp;
    __shared__ typename Min::TempStorage mintmp;
    __shared__ int l, r, s, bl, br, bs, winner, improved, lastbest, nbnd;
    __shared__ Score best;
    Counts local{};
    for (int u = threadIdx.x; u < n; u += NT) {
        if (saved[u] == 0)
            ++local.left;
        else if (saved[u] == 1)
            ++local.right;
        else
            ++local.sep;
    }
    Counts counts = Sum(sumtmp).Reduce(local, AddCounts{});
    __syncthreads();
    if (threadIdx.x == 0) {
        bl = counts.left;
        br = counts.right;
        bs = counts.sep;
        best = objective(bl, br, bs, total, balance);
    }
    __syncthreads();
    for (int pass = 0; pass < passes; ++pass) {
        int to = pass % 2, other = 1 - to;
        if (threadIdx.x == 0) {
            l = bl;
            r = br;
            s = bs;
            lastbest = -1;
            nbnd = 0;
        }
        __syncthreads();
        // Rebuild once per pass because saved labels may roll back exploratory
        // moves. Arbitrary packing order cannot affect the vertex-ID tie break.
        for (int u = threadIdx.x; u < n; u += NT) {
            current[u] = saved[u];
            bptr[u] = -1;
            if (current[u] == 2) {
                int pos = atomicAdd(&nbnd, 1);
                bnd[pos] = u;
                bptr[u] = pos;
            }
        }
        __syncthreads();
        for (int step = 0; step < moves; ++step) {
            Move candidate{{INT_MAX, INT_MAX, INT_MAX}, -1};
            int count = nbnd;
            for (int index = threadIdx.x; index < count; index += NT) {
                int u = bnd[index];
                if (current[u] != 2)
                    continue;
                int pulled = 0, w = 1;
                for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
                    int v = neighbor(g, e);
                    if (current[v] == other)
                        pulled += 1;
                }
                int nl = l + (to == 0 ? w : -pulled), nr = r + (to == 1 ? w : -pulled),
                    ns = s - w + pulled;
                candidate = MinMove{}(candidate, {objective(nl, nr, ns, total, balance), u});
            }
            Move chosen = Min(mintmp).Reduce(candidate, MinMove{});
            __syncthreads();
            if (threadIdx.x == 0) {
                winner = chosen.pos;
                improved = 0;
                if (winner >= 0) {
                    int w = 1, pulled = 0;
                    current[winner] = to;
                    // O(1) swap-remove, then append only newly pulled neighbors.
                    {
                        int pos = bptr[winner], last = bnd[--nbnd];
                        bnd[pos] = last;
                        bptr[last] = pos;
                        bptr[winner] = -1;
                    }
                    for (int e = g.row[winner]; e < g.row[winner + 1]; ++e) {
                        int v = neighbor(g, e);
                        if (current[v] == other) {
                            pulled += 1;
                            current[v] = 2;
                            bptr[v] = nbnd;
                            bnd[nbnd++] = v;
                        }
                    }
                    l += (to == 0 ? w : -pulled);
                    r += (to == 1 ? w : -pulled);
                    s += pulled - w;
                    Score score = objective(l, r, s, total, balance);
                    if (less(score, best)) {
                        best = score;
                        bl = l;
                        br = r;
                        bs = s;
                        improved = 1;
                        lastbest = step;
                    }
                }
            }
            __syncthreads();
            if (winner < 0)
                break;
            if (improved)
                for (int u = threadIdx.x; u < n; u += NT)
                    saved[u] = current[u];
            __syncthreads();
            if (step - lastbest >= stall)
                break;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[t] = {bl, br, bs, best};
}
__global__ void select_trial(int count, const Trial *trials, int *winner) {
    trials += size_t(blockIdx.x) * count;
    winner += blockIdx.x;
    using Reduce = cub::BlockReduce<Move, NT>;
    __shared__ typename Reduce::TempStorage tmp;
    Move best{{INT_MAX, INT_MAX, INT_MAX}, -1};
    for (int t = threadIdx.x; t < count; t += NT)
        best = MinMove{}(best, {trials[t].score, t});
    best = Reduce(tmp).Reduce(best, MinMove{});
    if (threadIdx.x == 0)
        *winner = best.pos;
}
__global__ void induced_degrees(View global, const int *ids, int m, int batches, int *degrees,
                                Node *local) {
    int i = blockIdx.x * NT + threadIdx.x;
    if (i > m * batches)
        return;
    if (i == m * batches) {
        degrees[i] = 0;
        return;
    }
    int b = i / m, r = i % m;
    Node t = global.nodes[ids[b]];
    int degree = 0;
    if (r < t.end - t.beg) {
        int u = global.p[t.beg + r];
        for (int e = global.row[u]; e < global.row[u + 1]; ++e) {
            int v = global.inv[global.col[e]];
            degree += v >= t.beg && v < t.end;
        }
    }
    degrees[i] = degree;
    if (r == 0) {
        Node out;
        out.beg = b * m;
        out.end = b * m + t.end - t.beg;
        out.origin = t.origin;
        out.components = 1;
        local[b] = out;
    }
}
__global__ void induced_edges(View global, const int *ids, int m, int batches, const int *row,
                              int *col) {
    int i = blockIdx.x * NT + threadIdx.x;
    if (i >= m * batches)
        return;
    int b = i / m, r = i % m;
    Node t = global.nodes[ids[b]];
    if (r >= t.end - t.beg)
        return;
    int u = global.p[t.beg + r], out = row[i];
    for (int e = global.row[u]; e < global.row[u + 1]; ++e) {
        int v = global.inv[global.col[e]];
        if (v >= t.beg && v < t.end)
            col[out++] = b * m + v - t.beg;
    }
}
__global__ void audit_trials(View g, int stride, const int *labels, const Trial *stats,
                             int *error) {
    int b = blockIdx.y, t = blockIdx.x;
    g = slice(g, b, stride);
    labels += size_t(b) * stride * gridDim.x + size_t(t) * stride;
    stats += b * gridDim.x + t;
    Counts local{};
    for (int u = threadIdx.x; u < vertices(g); u += NT) {
        int s = labels[u];
        if (s < 0 || s > 2) {
            atomicExch(error, 1);
            continue;
        }
        if (s == 0)
            ++local.left;
        else if (s == 1)
            ++local.right;
        else
            ++local.sep;
        for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
            int v = neighbor(g, e);
            if (v < 0 || v >= vertices(g)) {
                atomicExch(error, 2);
                continue;
            }
            if (s != 2 && labels[v] == 1 - s)
                atomicExch(error, 3);
        }
    }
    using Reduce = cub::BlockReduce<Counts, NT>;
    __shared__ typename Reduce::TempStorage tmp;
    Counts all = Reduce(tmp).Reduce(local, AddCounts{});
    if (threadIdx.x == 0 &&
        (all.left != stats->left || all.right != stats->right || all.sep != stats->sep))
        atomicExch(error, 4);
}
__global__ void publish(View global, const int *ids, int stride, int candidates, const int *winners,
                        const Trial *stats, const int *labels, int *output, float balance,
                        float maxsep) {
    int b = blockIdx.x, id = ids[b], winner = winners[b];
    Node t = global.nodes[id];
    Trial cut = stats[b * candidates + winner];
    bool valid = cut.left > 0 && cut.right > 0 &&
                 max(cut.left, cut.right) <= balance * (cut.left + cut.right) &&
                 cut.sep <= maxsep * (t.end - t.beg);
    __syncthreads();
    if (threadIdx.x == 0) {
        Node &node = global.nodes[id];
        node.split = valid;
        node.left = cut.left;
        node.right = cut.right;
        node.sep = cut.sep;
    }
    if (valid)
        for (int r = threadIdx.x; r < t.end - t.beg; r += NT)
            output[t.beg + r] = labels[(size_t(b) * candidates + winner) * stride + r];
}
__global__ void scan_flags(View g, const int *labels, U64 *flags) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.n)
        return;
    int id = g.owner[i];
    flags[i] = (id >= 0 && g.nodes[id].split)
                   ? (labels[i] == 0 ? U64(1) : (labels[i] == 1 ? (U64(1) << 32) : 0))
                   : 0;
}
__global__ void scatter(View g, const int *labels, const U64 *prefix, int *p, int *inv, int *owner,
                        int *final_owner) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.n)
        return;
    int id = g.owner[i], u = g.p[i], dest = i, new_owner = -1;
    if (id >= 0) {
        Node t = g.nodes[id];
        if (t.split) {
            U64 before = prefix[i] - prefix[t.beg];
            int l = unsigned(before), r = unsigned(before >> 32);
            if (labels[i] == 0) {
                dest = t.beg + l;
                new_owner = t.child;
            } else if (labels[i] == 1) {
                dest = t.beg + t.left + r;
                new_owner = t.child + 1;
            } else {
                dest = t.beg + t.left + t.right + i - t.beg - l - r;
                final_owner[u] = id;
            }
        } else
            final_owner[u] = id;
    }
    p[dest] = u;
    inv[u] = dest;
    owner[dest] = new_owner;
}
__global__ void children(Node *nodes, int first, int count, int next_first, bool terminal,
                         int *active) {
    int k = blockIdx.x * NT + threadIdx.x;
    if (k >= count)
        return;
    Node &t = nodes[first + k];
    if (terminal)
        t.split = 0;
    if (t.split) {
        int child = next_first + atomicAdd(active, 2);
        t.child = child;
        Node left, right;
        left.beg = t.beg;
        left.end = t.beg + t.left;
        left.origin = 2 * t.origin + 1;
        right.beg = left.end;
        right.end = right.beg + t.right;
        right.origin = 2 * t.origin + 2;
        nodes[child] = left;
        nodes[child + 1] = right;
        t.final_size = t.sep;
    } else
        t.final_size = t.end - t.beg;
}
__global__ void export_keys(int n, const int *p, const int *final_owner, const Node *nodes,
                            int levels, U64 *keys, int *vals, int *error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int u = p[i], id = final_owner[u];
    if (id < 0) {
        atomicMax(error, 1);
        return;
    }
    keys[i] = (U64(tree_rank(nodes[id].origin, levels)) << 32) | unsigned(i);
    vals[i] = u;
}
__global__ void export_parts(Node *nodes, int count, int levels, int *parts) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count)
        parts[tree_rank(nodes[id].origin, levels)] = nodes[id].final_size;
}
__global__ void invert(int n, const int *p, int *inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        inv[p[i]] = i;
}

__global__ void initialize_order(I n, I *p, I *inverse, I *owner, I *final_owner, Node *nodes) {
    I i = blockIdx.x * NT + threadIdx.x;
    if (i < n) {
        p[i] = inverse[i] = i;
        owner[i] = 0;
        final_owner[i] = -1;
    }
    if (i == 0) {
        nodes[0] = Node{};
        nodes[0].end = n;
    }
}
__global__ void collect_statistics(const Node *nodes, I count, I leaf, Statistics *stats) {
    I i = blockIdx.x * NT + threadIdx.x;
    if (i >= count)
        return;
    Node t = nodes[i];
    if (i == 0)
        stats->root_separator = t.split ? t.sep : 0;
    atomicMax(&stats->max_depth, 31 - __clz(t.origin + 1));
    if (t.split) {
        if (t.components > 1)
            atomicAdd(&stats->component_splits, 1);
    } else {
        atomicAdd(&stats->leaves, 1);
        atomicMax(&stats->largest_leaf, t.end - t.beg);
        if (t.end - t.beg > leaf)
            atomicAdd(&stats->oversized_leaves, 1);
    }
}
} // namespace

struct Ordering::Workspace {
    Graph input;
    Options options;
    cudaStream_t stream;
    Arena arena;
    I capacity, max_batch, candidates, max_bucket;
    I *p, *inverse, *owner, *next_p, *next_inverse, *next_owner, *final_owner;
    I *parents, *sizes, *roots, *component_prefix, *labels, *bin_counts, *bin_ids;
    Node *nodes;
    I *degree, *packed_row, *packed_col;
    Node *packed_nodes;
    I *trial_labels, *boundary, *boundary_position, *binary, *distance;
    unsigned char *work;
    I *values, *sorted_values, *offsets, *winners;
    U64 *keys, *sorted_keys, *flags, *prefix;
    Trial *trials;
    I *output_p, *output_inverse, *output_parts, *status;
    Statistics *device_statistics;
    Statistics stats{};
    void *scratch = nullptr;
    std::size_t scratch_bytes = 0;
    I *host = nullptr;

    Workspace(Graph graph, Options opts, cudaStream_t s) : input(graph), options(opts), stream(s) {
        if (input.n < 0 || input.nnz < 0 || !input.row_offsets ||
            (input.nnz && !input.column_indices))
            throw std::invalid_argument("invalid graph dimensions or device pointers");
        if (opts.trials < 1 || opts.trials > 128 || opts.leaf_size < 1 || opts.max_levels < 1 ||
            opts.max_levels > 22 || opts.refinement_passes < 0 || opts.moves_per_pass < 1 ||
            opts.stall_limit < 1 || !std::isfinite(opts.balance) || opts.balance < 0.5f ||
            opts.balance >= 1 || !std::isfinite(opts.max_separator_fraction) ||
            opts.max_separator_fraction <= 0 || opts.max_separator_fraction >= 1)
            throw std::invalid_argument("invalid ordering options");
        // Keep CUB item counts, packed offsets, and CUDA grid arithmetic in int32.
        auto slots = std::max<std::int64_t>(64, 2LL * input.n);
        if (slots * 2 * opts.trials > INT_MAX - NT || 32LL * input.n > INT_MAX)
            throw std::invalid_argument("workspace item counts exceed int32 range");
        capacity = I(slots);
        max_batch = std::max(1, I(input.n / (std::int64_t(opts.leaf_size) + 1)));
        candidates = 2 * opts.trials;
        max_bucket = 0;
        while ((1LL << max_bucket) < input.n)
            ++max_bucket;
        status = arena.ints(32);
        // Pinned scalar control storage avoids pageable async-copy staging.
        check(cudaMallocHost(reinterpret_cast<void **>(&host), 32 * sizeof(I)));
    }
    ~Workspace() {
        if (host) {
            cudaStreamSynchronize(stream);
            cudaFreeHost(host);
        }
    }

    void zero(void *p, std::size_t bytes) {
        if (bytes)
            check(cudaMemsetAsync(p, 0, bytes, stream));
    }
    I scalar(const I *p) {
        check(cudaMemcpyAsync(host, p, sizeof(I), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return host[0];
    }
    View graph() const {
        return {input.n, input.row_offsets, input.column_indices, p, inverse, owner, nodes};
    }
    void allocate() {
        zero(status, sizeof(I));
        validate_offsets<<<(input.n + 1 + NT - 1) / NT, NT, 0, stream>>>(input, status);
        if (scalar(status))
            throw std::invalid_argument("invalid CSR row offsets");
        if (input.n)
            validate_columns<<<(input.n + NT - 1) / NT, NT, 0, stream>>>(input, status);
        if (scalar(status))
            throw std::invalid_argument(
                "CSR must have sorted unique columns and a symmetric pattern");
        I n = input.n;
        p = arena.ints(n);
        inverse = arena.ints(n);
        owner = arena.ints(n);
        next_p = arena.ints(n);
        next_inverse = arena.ints(n);
        next_owner = arena.ints(n);
        final_owner = arena.ints(n);
        parents = arena.ints(n);
        sizes = arena.ints(n);
        roots = arena.ints(n);
        component_prefix = arena.ints(n);
        labels = arena.ints(n);
        bin_counts = arena.ints(32);
        bin_ids = arena.ints(std::size_t(32) * max_batch);
        nodes = static_cast<Node *>(arena.bytes(std::size_t(std::max(1, 2 * n)) * sizeof(Node)));
        degree = arena.ints(capacity + 1);
        packed_row = arena.ints(capacity + 1);
        packed_col = arena.ints(input.nnz);
        packed_nodes = static_cast<Node *>(arena.bytes(std::size_t(max_batch) * sizeof(Node)));
        std::size_t trial_slots = std::size_t(capacity) * options.trials,
                    candidate_slots = 2 * trial_slots;
        trial_labels = arena.ints(candidate_slots);
        boundary = arena.ints(candidate_slots);
        boundary_position = arena.ints(candidate_slots);
        work = static_cast<unsigned char *>(arena.bytes(candidate_slots));
        binary = arena.ints(trial_slots);
        distance = arena.ints(trial_slots);
        values = arena.ints(trial_slots);
        sorted_values = arena.ints(trial_slots);
        offsets = arena.ints(std::size_t(max_batch) * options.trials + 1);
        winners = arena.ints(max_batch);
        keys = arena.keys(trial_slots);
        sorted_keys = arena.keys(trial_slots);
        flags = arena.keys(n);
        prefix = arena.keys(n);
        trials =
            static_cast<Trial *>(arena.bytes(std::size_t(max_batch) * candidates * sizeof(Trial)));
        output_p = arena.ints(n);
        output_inverse = arena.ints(n);
        output_parts = arena.ints((1 << options.max_levels) - 1);
        device_statistics = static_cast<Statistics *>(arena.bytes(sizeof(Statistics)));
        std::size_t bytes = 0;
        check(cub::DeviceScan::InclusiveSum(nullptr, bytes, sizes, component_prefix, std::max(1, n),
                                            stream));
        scratch_bytes = std::max(scratch_bytes, bytes);
        check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, degree, packed_row, capacity + 1,
                                            stream));
        scratch_bytes = std::max(scratch_bytes, bytes);
        check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, flags, prefix, std::max(1, n), stream));
        scratch_bytes = std::max(scratch_bytes, bytes);
        check(cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, bytes, keys, sorted_keys, values, sorted_values, I(trial_slots),
            max_batch * options.trials, offsets, offsets + 1, 0, 64, stream));
        scratch_bytes = std::max(scratch_bytes, bytes);
        check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, sorted_keys, values, output_p,
                                              std::max(1, n), 0, 64, stream));
        scratch_bytes = std::max(scratch_bytes, bytes);
        scratch = arena.bytes(scratch_bytes);
        check(cudaGetLastError());
        check(cudaStreamSynchronize(stream));
    }

    void bisect(const I *ids, I stride, I batch) {
        if (std::int64_t(stride) * batch > capacity || batch > max_batch)
            throw std::runtime_error("partition workspace capacity exceeded");
        I count = stride * batch;
        induced_degrees<<<(count + 1 + NT - 1) / NT, NT, 0, stream>>>(graph(), ids, stride, batch,
                                                                      degree, packed_nodes);
        std::size_t bytes = scratch_bytes;
        check(cub::DeviceScan::ExclusiveSum(scratch, bytes, degree, packed_row, count + 1, stream));
        induced_edges<<<(count + NT - 1) / NT, NT, 0, stream>>>(graph(), ids, stride, batch,
                                                                packed_row, packed_col);
        View packed{count, packed_row, packed_col, nullptr, nullptr, nullptr, packed_nodes};
        I num_trials = options.trials;
        grow<<<dim3(num_trials, batch), NT, 0, stream>>>(packed, num_trials, options.seed, distance,
                                                         stride);
        bfs_keys<<<dim3((stride * num_trials + NT - 1) / NT, batch), NT, 0, stream>>>(
            packed, num_trials, options.seed, distance, keys, values, offsets, stride);
        bytes = scratch_bytes;
        check(cub::DeviceSegmentedRadixSort::SortPairs(
            scratch, bytes, keys, sorted_keys, values, sorted_values, count * num_trials,
            batch * num_trials, offsets, offsets + 1, 0, 64, stream));
        split_half<<<dim3((stride * num_trials + NT - 1) / NT, batch), NT, 0, stream>>>(
            packed, num_trials, sorted_values, binary, stride);
        covers<<<dim3((stride * candidates + NT - 1) / NT, batch), NT, 0, stream>>>(
            packed, candidates, binary, trial_labels, stride);
        std::size_t shared_bytes = stride <= 32768 ? stride : 0;
        refine<<<dim3(candidates, batch), NT, shared_bytes, stream>>>(
            packed, trial_labels, work, boundary, boundary_position, trials,
            options.refinement_passes, options.moves_per_pass, options.stall_limit, options.balance,
            stride);
        audit_trials<<<dim3(candidates, batch), NT, 0, stream>>>(packed, stride, trial_labels,
                                                                 trials, status + 1);
        select_trial<<<batch, NT, 0, stream>>>(candidates, trials, winners);
        publish<<<batch, NT, 0, stream>>>(graph(), ids, stride, candidates, winners, trials,
                                          trial_labels, labels, options.balance,
                                          options.max_separator_fraction);
        stats.peak_batch = std::max(stats.peak_batch, batch);
    }

    Order compute() {
        stats = {};
        zero(status, 32 * sizeof(I));
        I n = input.n;
        initialize_order<<<std::max(1, (n + NT - 1) / NT), NT, 0, stream>>>(n, p, inverse, owner,
                                                                            final_owner, nodes);
        I first = 0, count = 1, used = 1, levels = 1;
        if (n)
            for (I depth = 0; depth < options.max_levels; ++depth) {
                levels = depth + 1;
                bool terminal = levels == options.max_levels;
                if (!terminal) {
                    init_components<<<(n + NT - 1) / NT, NT, 0, stream>>>(n, parents, sizes);
                    join_components<<<(n + NT - 1) / NT, NT, 0, stream>>>(graph(), parents);
                    count_components<<<(n + NT - 1) / NT, NT, 0, stream>>>(graph(), parents, roots,
                                                                           sizes);
                    std::size_t bytes = scratch_bytes;
                    check(cub::DeviceScan::InclusiveSum(scratch, bytes, sizes, component_prefix, n,
                                                        stream));
                    component_cut<<<(n + NT - 1) / NT, NT, 0, stream>>>(
                        graph(), sizes, component_prefix, options.leaf_size);
                    component_labels<<<(n + NT - 1) / NT, NT, 0, stream>>>(
                        graph(), roots, component_prefix, labels, options.leaf_size);
                    zero(bin_counts, 32 * sizeof(I));
                    make_bins<<<(count + NT - 1) / NT, NT, 0, stream>>>(
                        nodes, first, count, options.leaf_size, bin_counts, bin_ids, max_batch);
                    check(cudaMemcpyAsync(host, bin_counts, 32 * sizeof(I), cudaMemcpyDeviceToHost,
                                          stream));
                    check(cudaStreamSynchronize(stream));
                    for (I bucket = 1; bucket <= max_bucket; ++bucket) {
                        // CUDA grid.y is limited to 65535; chunk larger buckets.
                        for (I start = 0; start < host[bucket]; start += 65535)
                            bisect(bin_ids + bucket * max_batch + start, 1 << bucket,
                                   std::min(65535, host[bucket] - start));
                    }
                }
                zero(status, sizeof(I));
                children<<<(count + NT - 1) / NT, NT, 0, stream>>>(nodes, first, count, used,
                                                                   terminal, status);
                scan_flags<<<(n + NT - 1) / NT, NT, 0, stream>>>(graph(), labels, flags);
                std::size_t bytes = scratch_bytes;
                check(cub::DeviceScan::ExclusiveSum(scratch, bytes, flags, prefix, n, stream));
                scatter<<<(n + NT - 1) / NT, NT, 0, stream>>>(
                    graph(), labels, prefix, next_p, next_inverse, next_owner, final_owner);
                std::swap(p, next_p);
                std::swap(inverse, next_inverse);
                std::swap(owner, next_owner);
                count = scalar(status);
                first = used;
                used += count;
                if (!count)
                    break;
            }
        I num_parts = (1 << levels) - 1;
        zero(output_parts, std::size_t(num_parts) * sizeof(I));
        export_parts<<<(used + NT - 1) / NT, NT, 0, stream>>>(nodes, used, levels, output_parts);
        if (n) {
            export_keys<<<(n + NT - 1) / NT, NT, 0, stream>>>(n, p, final_owner, nodes, levels,
                                                              keys, values, status + 1);
            std::size_t bytes = scratch_bytes;
            check(cub::DeviceRadixSort::SortPairs(scratch, bytes, keys, sorted_keys, values,
                                                  output_p, n, 0, 64, stream));
            invert<<<(n + NT - 1) / NT, NT, 0, stream>>>(n, output_p, output_inverse);
        }
        zero(device_statistics, sizeof(Statistics));
        collect_statistics<<<(used + NT - 1) / NT, NT, 0, stream>>>(nodes, used, options.leaf_size,
                                                                    device_statistics);
        check(cudaGetLastError());
        if (scalar(status + 1))
            throw std::runtime_error("GPU separator/export invariant failed");
        I peak = stats.peak_batch;
        check(cudaMemcpyAsync(&stats, device_statistics, sizeof(Statistics), cudaMemcpyDeviceToHost,
                              stream));
        check(cudaStreamSynchronize(stream));
        stats.peak_batch = peak;
        return {n, output_p, output_inverse, output_parts, num_parts, levels};
    }
};

Ordering::Ordering(Graph graph, Options options, cudaStream_t stream) : workspace_(nullptr) {
    auto workspace = std::make_unique<Workspace>(graph, options, stream);
    workspace->allocate();
    workspace_ = workspace.release();
}
Ordering::~Ordering() { delete workspace_; }
Order Ordering::compute() { return workspace_->compute(); }
Statistics Ordering::statistics() const { return workspace_->stats; }
} // namespace ndgpu

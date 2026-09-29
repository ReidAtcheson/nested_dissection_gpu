#include "support.hpp"
#include <iostream>
#include <random>

static void check_graph(const HostGraph &graph, int leaf = 8, int trials = 4, int passes = 4) {
    DeviceGraph device(graph);
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
        ndgpu::Options options;
        options.leaf_size = leaf;
        options.trials = trials;
        options.refinement_passes = passes;
        options.moves_per_pass = 32;
        options.max_levels = 16;
        ndgpu::Ordering ordering(device.view, options, stream);
        HostOrder a(ordering.compute());
        validate(graph, a);
        HostOrder b(ordering.compute());
        validate(graph, b);
        require(a.p == b.p && a.parts == b.parts, "nondeterministic replay");
        auto stats = ordering.statistics();
        require(stats.leaves > 0, "missing leaves");
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    cuda_check(cudaStreamDestroy(stream));
}
static void invalid(const HostGraph &graph) {
    DeviceGraph device(graph);
    bool rejected = false;
    try {
        ndgpu::Ordering ordering(device.view);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "invalid CSR was accepted");
}
// FNV-1a over the little-endian int32 values, independent of host byte order.
static std::uint64_t fingerprint(const std::vector<int> &values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned value : values)
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= (value >> shift) & 255u;
            hash *= 1099511628211ULL;
        }
    return hash;
}
static void reference_grids() {
    // Frozen outputs of the original research experiment, before this port.
    for (int side : {32, 64}) {
        auto g = grid(side);
        DeviceGraph device(g);
        ndgpu::Ordering ordering(device.view);
        HostOrder order(ordering.compute());
        validate(g, order);
        auto p = side == 32 ? 0x6d7c9fb3f1031585ULL : 0x58e4132d50657d15ULL;
        auto parts = side == 32 ? 0x2cbecb24280212eaULL : 0x02f00b26a132a47aULL;
        require(fingerprint(order.p) == p, "reference permutation changed");
        require(fingerprint(order.parts) == parts, "reference partition tree changed");
        require(ordering.statistics().oversized_leaves == 0, "reference has oversized leaves");
        std::cout << "PASS: " << side << "^3 matches the reference permutation and tree\n";
    }
}
int main(int argc, char **argv) try {
    cuda_check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    if (argc == 2 && std::string(argv[1]) == "--reference-grids") {
        reference_grids();
        return 0;
    }
    if (argc != 1)
        throw std::invalid_argument("usage: ndgpu_test [--reference-grids]");
    for (int side : {2, 3, 5, 7, 11})
        check_graph(grid(side));
    check_graph(grid(5), 8, 1, 0);
    check_graph(from_edges(0, {}));
    check_graph(from_edges(1, {}));
    check_graph(from_edges(97, {}));
    std::vector<std::pair<int, int>> edges;
    for (int i = 1; i < 97; ++i)
        edges.emplace_back(i - 1, i);
    check_graph(from_edges(97, edges), 1);
    edges.clear();
    for (int i = 1; i < 97; ++i)
        edges.emplace_back(0, i);
    check_graph(from_edges(97, edges));
    edges.clear();
    for (int i = 0; i < 40; ++i)
        for (int j = i + 1; j < 40; ++j)
            edges.emplace_back(i, j);
    check_graph(from_edges(40, edges)); // no acceptable balanced separator: oversized leaf
    edges.clear();
    std::mt19937 rng(7821);
    for (int i = 0; i < 161; ++i)
        for (int j = i + 1; j < 161; ++j)
            if (rng() % 61 == 0)
                edges.emplace_back(i, j);
    check_graph(from_edges(173, edges)); // irregular graph with isolates
    auto diagonal = grid(5);
    edges.clear();
    for (int u = 0; u < diagonal.size(); ++u) {
        edges.emplace_back(u, u);
        for (int e = diagonal.row[u]; e < diagonal.row[u + 1]; ++e)
            edges.emplace_back(u, diagonal.col[e]);
    }
    check_graph(from_edges(125, edges));
    invalid({{0, 2, 1}, {1}});
    invalid({{0, 1, 1}, {2}});
    invalid({{0, 1, 1}, {1}});
    invalid({{0, 2, 4}, {1, 1, 0, 0}});
    invalid({{0, 2, 4}, {1, 0, 0, 1}});
    DeviceGraph small(grid(2));
    ndgpu::Options shallow;
    shallow.max_levels = 1;
    shallow.leaf_size = 1;
    ndgpu::Ordering limited(small.view, shallow);
    HostOrder o(limited.compute());
    validate(grid(2), o);
    require(limited.statistics().oversized_leaves == 1, "depth cap not reported");
    ndgpu::Options bad;
    bad.trials = 0;
    bool rejected = false;
    try {
        ndgpu::Ordering test(small.view, bad);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "invalid options accepted");
    std::cout << "PASS: grid/irregular/disconnected CSR, replay, nondefault stream, input errors "
                 "and depth cap\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}

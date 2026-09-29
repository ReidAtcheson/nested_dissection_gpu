#pragma once
#include "ndgpu/ordering.hpp"
#include <algorithm>
#include <fstream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

inline void cuda_check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
struct HostGraph {
    std::vector<std::int32_t> row, col;
    int size() const { return int(row.size()) - 1; }
};
inline HostGraph from_edges(int n, const std::vector<std::pair<int, int>> &edges) {
    std::vector<std::set<int>> adjacency(n);
    for (auto [u, v] : edges) {
        adjacency[u].insert(v);
        adjacency[v].insert(u);
    }
    HostGraph g;
    g.row.push_back(0);
    for (auto &row : adjacency) {
        g.col.insert(g.col.end(), row.begin(), row.end());
        g.row.push_back(g.col.size());
    }
    return g;
}
inline HostGraph grid(int side) {
    if (side < 1 || side > 128)
        throw std::invalid_argument("grid side must be in 1..128");
    std::vector<std::pair<int, int>> edges;
    for (int z = 0; z < side; ++z)
        for (int y = 0; y < side; ++y)
            for (int x = 0; x < side; ++x) {
                int u = x + side * (y + side * z);
                if (x + 1 < side)
                    edges.emplace_back(u, u + 1);
                if (y + 1 < side)
                    edges.emplace_back(u, u + side);
                if (z + 1 < side)
                    edges.emplace_back(u, u + side * side);
            }
    return from_edges(side * side * side, edges);
}
struct DeviceGraph {
    std::int32_t *row = nullptr, *col = nullptr;
    ndgpu::Graph view{};
    explicit DeviceGraph(const HostGraph &g) {
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&row), g.row.size() * sizeof(int)));
        try {
            if (!g.col.empty())
                cuda_check(cudaMalloc(reinterpret_cast<void **>(&col), g.col.size() * sizeof(int)));
            cuda_check(
                cudaMemcpy(row, g.row.data(), g.row.size() * sizeof(int), cudaMemcpyHostToDevice));
            if (!g.col.empty())
                cuda_check(cudaMemcpy(col, g.col.data(), g.col.size() * sizeof(int),
                                      cudaMemcpyHostToDevice));
        } catch (...) {
            cudaFree(row);
            cudaFree(col);
            throw;
        }
        view = {g.size(), int(g.col.size()), row, col};
    }
    ~DeviceGraph() {
        cudaFree(row);
        cudaFree(col);
    }
    DeviceGraph(const DeviceGraph &) = delete;
    DeviceGraph &operator=(const DeviceGraph &) = delete;
};
inline std::vector<int> download(const int *p, int n) {
    std::vector<int> out(n);
    if (n)
        cuda_check(cudaMemcpy(out.data(), p, std::size_t(n) * sizeof(int), cudaMemcpyDeviceToHost));
    return out;
}
struct HostOrder {
    std::vector<int> p, inverse, parts;
    int levels = 0;
    HostOrder() = default;
    explicit HostOrder(ndgpu::Order o)
        : p(download(o.permutation, o.n)), inverse(download(o.inverse, o.n)),
          parts(download(o.parts, o.num_parts)), levels(o.num_levels) {}
};
inline void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
inline void validate(const HostGraph &g, const HostOrder &order) {
    int n = g.size();
    require(int(order.p.size()) == n, "permutation size");
    require(order.levels >= 1 && order.levels <= 22 &&
                int(order.parts.size()) == (1 << order.levels) - 1,
            "tree shape");
    std::vector<int> seen(n), owner(n, -1), by_rank(order.parts.size());
    for (int k = 0; k < n; ++k) {
        int u = order.p[k];
        require(u >= 0 && u < n && !seen[u]++, "permutation bijection");
        require(order.inverse[u] == k, "inverse mismatch");
    }
    for (int id = 0; id < int(order.parts.size()); ++id) {
        int depth = 0;
        for (int v = id + 1; v > 1; v >>= 1)
            ++depth;
        int rank = (1 << order.levels) - (1 << (depth + 1)) + id - ((1 << depth) - 1);
        by_rank[rank] = id;
    }
    int offset = 0;
    for (int k = 0; k < int(order.parts.size()); ++k) {
        int count = order.parts[k];
        require(count >= 0 && count <= n - offset, "tree sizes");
        for (int j = 0; j < count; ++j)
            owner[order.p[offset++]] = by_rank[k];
    }
    require(offset == n, "tree total");
    auto ancestor = [](int a, int b) {
        while (b > a)
            b = (b - 1) / 2;
        return a == b;
    };
    for (int u = 0; u < n; ++u)
        for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
            int a = owner[u], b = owner[g.col[e]];
            require(ancestor(a, b) || ancestor(b, a), "edge crosses independent tree branches");
        }
}
inline void save(const std::string &prefix, const HostOrder &o) {
    for (auto entry : {std::make_pair(".p.txt", &o.p), std::make_pair(".invp.txt", &o.inverse),
                       std::make_pair(".parts.txt", &o.parts)}) {
        std::ofstream f(prefix + entry.first);
        if (!f)
            throw std::runtime_error("cannot open output");
        for (int x : *entry.second)
            f << x << '\n';
        if (!f)
            throw std::runtime_error("output write failed");
    }
}

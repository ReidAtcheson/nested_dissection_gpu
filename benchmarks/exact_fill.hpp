#pragma once
#include "../tests/support.hpp"
#include <climits>
#include <cs.h>
#include <cstdint>

// Exact scalar Cholesky counts: compare etree reach against CXSparse's
// independent symbolic column-count algorithm. Neither forms the filled graph.
inline std::uint64_t exact_fill(const HostGraph &graph, const std::vector<int> &permutation) {
    int n = graph.size();
    require(int(permutation.size()) == n, "fill permutation size");
    std::vector<int> inverse(n, -1);
    for (int k = 0; k < n; ++k) {
        int u = permutation[k];
        require(u >= 0 && u < n && inverse[u] < 0, "fill permutation invalid");
        inverse[u] = k;
    }
    std::vector<std::vector<int>> upper(n);
    for (int k = 0; k < n; ++k)
        upper[k].push_back(k);
    for (int u = 0; u < n; ++u)
        for (int e = graph.row[u]; e < graph.row[u + 1]; ++e)
            if (u < graph.col[e]) {
                int i = inverse[u], j = inverse[graph.col[e]];
                if (i > j)
                    std::swap(i, j);
                upper[j].push_back(i);
            }
    std::size_t nnz = 0;
    for (auto &column : upper) {
        std::sort(column.begin(), column.end());
        nnz += column.size();
    }
    require(nnz <= INT_MAX, "symbolic input exceeds int32 nnz");
    std::vector<int> parent(n, -1), ancestor(n, -1), count(n, 1), mark(n, -1);
    for (int k = 0; k < n; ++k)
        for (int j : upper[k])
            if (j < k)
                while (j != -1 && j < k) {
                    int next = ancestor[j];
                    ancestor[j] = k;
                    if (next == -1)
                        parent[j] = k;
                    j = next;
                }
    for (int k = 0; k < n; ++k) {
        mark[k] = k;
        for (int j : upper[k])
            if (j < k)
                while (j != -1 && mark[j] != k) {
                    ++count[j];
                    mark[j] = k;
                    j = parent[j];
                }
    }
    struct Symbolic {
        cs_di *a = nullptr;
        int *parent = nullptr, *post = nullptr, *counts = nullptr;
        ~Symbolic() {
            cs_di_free(counts);
            cs_di_free(post);
            cs_di_free(parent);
            cs_di_spfree(a);
        }
    } check;
    check.a = cs_di_spalloc(n, n, int(nnz), 0, 0);
    require(check.a, "CXSparse allocation failed");
    int offset = 0;
    for (int k = 0; k < n; ++k) {
        check.a->p[k] = offset;
        for (int i : upper[k])
            check.a->i[offset++] = i;
    }
    check.a->p[n] = offset;
    check.parent = cs_di_etree(check.a, 0);
    require(check.parent, "CXSparse etree failed");
    check.post = cs_di_post(check.parent, n);
    require(check.post, "CXSparse postorder failed");
    check.counts = cs_di_counts(check.a, check.parent, check.post, 0);
    require(check.counts, "CXSparse counts failed");
    std::uint64_t total = 0;
    for (int k = 0; k < n; ++k) {
        require(count[k] == check.counts[k], "independent exact column counts disagree");
        total += count[k];
    }
    return total;
}

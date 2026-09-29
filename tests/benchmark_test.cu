#include "../benchmarks/exact_fill.hpp"
#include "../benchmarks/matrix_market.hpp"
#include <cstdio>
#include <iostream>
#include <random>
#include <unistd.h>

static std::uint64_t explicit_fill(const HostGraph &g, const std::vector<int> &p) {
    std::vector<int> inverse(p.size());
    for (int k = 0; k < int(p.size()); ++k)
        inverse[p[k]] = k;
    std::vector<std::set<int>> adjacency(p.size());
    for (int u = 0; u < g.size(); ++u)
        for (int e = g.row[u]; e < g.row[u + 1]; ++e) {
            int a = inverse[u], b = inverse[g.col[e]];
            if (a < b)
                adjacency[a].insert(b);
        }
    std::uint64_t total = g.size();
    for (int k = 0; k < g.size(); ++k) {
        total += adjacency[k].size();
        for (int a : adjacency[k])
            for (int b : adjacency[k])
                if (a < b)
                    adjacency[a].insert(b);
    }
    return total;
}
int main() try {
    std::mt19937 rng(912);
    for (int side : {2, 3, 4}) {
        auto g = grid(side);
        std::vector<int> p(g.size());
        std::iota(p.begin(), p.end(), 0);
        for (int rep = 0; rep < 8; ++rep) {
            std::shuffle(p.begin(), p.end(), rng);
            require(exact_fill(g, p) == explicit_fill(g, p),
                    "fill counter differs from explicit elimination");
        }
    }
    struct Temporary {
        std::string path;
        ~Temporary() { std::remove(path.c_str()); }
    } file{"/tmp/ndgpu-matrix-test-" + std::to_string(getpid()) + ".mtx"};
    {
        std::ofstream f(file.path);
        f << "%%MatrixMarket matrix coordinate real symmetric\n% test\n3 3 5\n1 1 1\n2 1 0\n3 2 "
             "1\n3 2 2\n3 3 1\n";
    }
    auto g = read_matrix_market(file.path);
    require(g.row == std::vector<int>({0, 1, 3, 4}) && g.col == std::vector<int>({1, 0, 2, 1}),
            "zero/duplicate/symmetry parsing");
    require(exact_fill(g, {0, 1, 2}) == 5 && exact_fill(g, {1, 0, 2}) == 6, "path symbolic fill");
    for (auto text : {"%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 2\n",
                      "%%MatrixMarket matrix coordinate real symmetric\n2 2 1\n1 3 1\n",
                      "%%MatrixMarket matrix coordinate real symmetric\n2 2 1\n1 2\n"}) {
        {
            std::ofstream f(file.path);
            f << text;
        }
        bool rejected = false;
        try {
            read_matrix_market(file.path);
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "invalid matrix accepted");
    }
    std::cout << "PASS: Matrix Market canonicalization/errors and exact fill against explicit "
                 "elimination\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}

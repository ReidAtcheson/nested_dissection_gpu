#pragma once
#include "../tests/support.hpp"
#include <cctype>
#include <climits>
#include <sstream>

// Structural input: retain stored entries (including explicit zero values),
// expand symmetric storage, deduplicate, and omit the diagonal from the graph.
inline HostGraph read_matrix_market(const std::string &path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open Matrix Market file: " + path);
    std::string line, banner, object, format, field, symmetry;
    if (!std::getline(input, line))
        throw std::runtime_error("missing Matrix Market banner");
    std::istringstream header(line);
    header >> banner >> object >> format >> field >> symmetry;
    for (auto *token : {&banner, &object, &format, &field, &symmetry})
        std::transform(token->begin(), token->end(), token->begin(),
                       [](unsigned char c) { return std::tolower(c); });
    require(banner == "%%matrixmarket" && object == "matrix" && format == "coordinate",
            "expected coordinate Matrix Market input");
    require(field == "real" || field == "integer" || field == "pattern" || field == "complex",
            "unsupported Matrix Market field");
    require(symmetry == "symmetric" || symmetry == "hermitian" || symmetry == "general",
            "unsupported Matrix Market symmetry");
    do {
        if (!std::getline(input, line))
            throw std::runtime_error("missing matrix dimensions");
    } while (line.empty() || line[0] == '%');
    std::int64_t rows = 0, cols = 0, entries = 0;
    std::istringstream dimensions(line);
    require(bool(dimensions >> rows >> cols >> entries), "invalid matrix dimensions");
    require(rows > 0 && rows == cols && rows <= INT_MAX && entries >= 0 && entries <= INT_MAX / 2,
            "matrix exceeds int32 dimensions or is not square");
    std::vector<std::pair<int, int>> edges;
    edges.reserve(std::size_t(entries) * (symmetry == "general" ? 1 : 2));
    for (std::int64_t k = 0; k < entries; ++k) {
        std::int64_t u, v;
        double real = 0, imag = 0;
        require(bool(input >> u >> v), "truncated Matrix Market coordinates");
        if (field != "pattern")
            require(bool(input >> real), "invalid Matrix Market value");
        if (field == "complex")
            require(bool(input >> imag), "invalid complex Matrix Market value");
        require(u >= 1 && u <= rows && v >= 1 && v <= rows, "Matrix Market index out of range");
        if (u == v)
            continue;
        edges.emplace_back(int(u - 1), int(v - 1));
        if (symmetry != "general")
            edges.emplace_back(int(v - 1), int(u - 1));
    }
    std::string extra;
    require(!(input >> extra), "extra Matrix Market data");
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    if (symmetry == "general")
        for (auto [u, v] : edges)
            require(std::binary_search(edges.begin(), edges.end(), std::make_pair(v, u)),
                    "general matrix pattern is not symmetric");
    require(edges.size() <= INT_MAX, "graph exceeds int32 nnz");
    HostGraph graph;
    graph.row.resize(rows + 1);
    graph.col.reserve(edges.size());
    for (auto [u, v] : edges) {
        ++graph.row[u + 1];
        graph.col.push_back(v);
    }
    std::partial_sum(graph.row.begin(), graph.row.end(), graph.row.begin());
    return graph;
}

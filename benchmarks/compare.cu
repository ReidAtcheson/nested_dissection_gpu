#include "exact_fill.hpp"
#include "matrix_market.hpp"
#include <chrono>
#include <cstdio>
#include <ctime>
#include <cudss.h>
#include <iostream>

static double cpu_ms() {
    timespec t{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec * 1000. + t.tv_nsec * 1e-6;
}
static void dss_check(cudssStatus_t status) {
    if (status != CUDSS_STATUS_SUCCESS)
        throw std::runtime_error("cuDSS status " + std::to_string(int(status)));
}
struct Context {
    cudssHandle_t handle{};
    cudssConfig_t config{};
    cudssData_t data{};
    ~Context() {
        if (data)
            cudssDataDestroy(handle, data);
        if (config)
            cudssConfigDestroy(config);
        if (handle)
            cudssDestroy(handle);
    }
    void create() {
        dss_check(cudssCreate(&handle));
        dss_check(cudssConfigCreate(&config));
        dss_check(cudssDataCreate(handle, &data));
        auto alg = CUDSS_REORDERING_ALG_NESTED_DISSECTION;
        dss_check(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)));
    }
    std::vector<int> array(cudssDataParam_t key) {
        size_t bytes = 0, written = 0;
        dss_check(cudssDataGet(handle, data, key, nullptr, 0, &bytes));
        std::vector<int> result(bytes / sizeof(int));
        dss_check(cudssDataGet(handle, data, key, result.data(), bytes, &written));
        require(bytes == written, "cuDSS array size");
        return result;
    }
};
struct Matrix {
    int *row = nullptr, *col = nullptr;
    double *val = nullptr, *b = nullptr, *x = nullptr;
    cudssMatrix_t a{}, rhs{}, solution{};
    ~Matrix() {
        if (a)
            cudssMatrixDestroy(a);
        if (rhs)
            cudssMatrixDestroy(rhs);
        if (solution)
            cudssMatrixDestroy(solution);
        cudaFree(row);
        cudaFree(col);
        cudaFree(val);
        cudaFree(b);
        cudaFree(x);
    }
    void create(const HostGraph &g) {
        // Graph Laplacian + I: same stored off-diagonal pattern, strictly SPD.
        // Only reordering is executed; numerical values do not enter fill counts.
        int n = g.size();
        std::vector<int> r(n + 1), c;
        std::vector<double> v;
        for (int u = 0; u < n; ++u) {
            for (int e = g.row[u]; e < g.row[u + 1]; ++e)
                if (g.col[e] < u) {
                    c.push_back(g.col[e]);
                    v.push_back(-1);
                }
            c.push_back(u);
            v.push_back(g.row[u + 1] - g.row[u] + 1);
            r[u + 1] = c.size();
        }
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&row), r.size() * sizeof(int)));
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&col), c.size() * sizeof(int)));
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&val), v.size() * sizeof(double)));
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&b), std::size_t(n) * sizeof(double)));
        cuda_check(cudaMalloc(reinterpret_cast<void **>(&x), std::size_t(n) * sizeof(double)));
        cuda_check(cudaMemcpy(row, r.data(), r.size() * sizeof(int), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(col, c.data(), c.size() * sizeof(int), cudaMemcpyHostToDevice));
        cuda_check(cudaMemcpy(val, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice));
        cuda_check(cudaMemset(b, 0, std::size_t(n) * sizeof(double)));
        cuda_check(cudaMemset(x, 0, std::size_t(n) * sizeof(double)));
        dss_check(cudssMatrixCreateCsr(&a, n, n, c.size(), row, nullptr, col, val, CUDSS_R_32I,
                                       CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER,
                                       CUDSS_BASE_ZERO));
        dss_check(cudssMatrixCreateDn(&rhs, n, 1, n, b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        dss_check(cudssMatrixCreateDn(&solution, n, 1, n, x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    }
};
int main(int argc, char **argv) try {
    if (argc < 4 || argc > 6)
        throw std::invalid_argument(
            "usage: ndgpu_compare MATRIX.mtx|grid:SIDE NAME REPS [OUTPUT_PREFIX] [bfs|cudss|both]");
    std::string input = argv[1], name = argv[2], mode = argc > 5 ? argv[5] : "both";
    int reps = std::stoi(argv[3]);
    require(reps > 0, "reps must be positive");
    require(mode == "both" || mode == "bfs" || mode == "cudss", "invalid mode");
    cuda_check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    auto graph =
        input.rfind("grid:", 0) == 0 ? grid(std::stoi(input.substr(5))) : read_matrix_market(input);
    int n = graph.size();
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, 0));
    int major, minor, patch;
    dss_check(cudssGetProperty(MAJOR_VERSION, &major));
    dss_check(cudssGetProperty(MINOR_VERSION, &minor));
    dss_check(cudssGetProperty(PATCH_LEVEL, &patch));
    std::fprintf(stderr, "matrix=%s n=%d graph_nnz=%zu gpu=%s cudss=%d.%d.%d\n", name.c_str(), n,
                 graph.col.size(), properties.name, major, minor, patch);
    std::puts("matrix,n,graph_nnz,method,sample,wall_ms,cpu_ms,exact_nnz_l,levels,leaves,largest_"
              "leaf,oversized_leaves,peak_batch");
    if (mode != "cudss") {
        DeviceGraph device(graph);
        ndgpu::Ordering ordering(device.view);
        std::vector<int> previous;
        std::uint64_t fill = 0;
        for (int rep = -1; rep < reps; ++rep) {
            cuda_check(cudaDeviceSynchronize());
            auto start = std::chrono::steady_clock::now();
            double cpu = cpu_ms();
            auto order = ordering.compute();
            cpu = cpu_ms() - cpu;
            double wall =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            HostOrder host(order);
            validate(graph, host);
            auto stats = ordering.statistics();
            if (rep >= 0) {
                if (host.p != previous) {
                    fill = exact_fill(graph, host.p);
                    previous = host.p;
                }
                std::printf("%s,%d,%zu,bfs,%d,%.6f,%.6f,%llu,%d,%d,%d,%d,%d\n", name.c_str(), n,
                            graph.col.size(), rep, wall, cpu, (unsigned long long)fill, host.levels,
                            stats.leaves, stats.largest_leaf, stats.oversized_leaves,
                            stats.peak_batch);
                std::fflush(stdout);
            }
            if (rep == reps - 1 && argc > 4)
                save(std::string(argv[4]) + ".bfs", host);
        }
    }
    if (mode != "bfs") {
        Matrix matrix;
        matrix.create(graph);
        std::vector<int> previous;
        std::uint64_t fill = 0;
        for (int rep = -1; rep < reps; ++rep) {
            Context context;
            context.create();
            cuda_check(cudaDeviceSynchronize());
            auto start = std::chrono::steady_clock::now();
            double cpu = cpu_ms();
            dss_check(cudssExecute(context.handle, CUDSS_PHASE_REORDERING, context.config,
                                   context.data, matrix.a, matrix.solution, matrix.rhs));
            cuda_check(cudaDeviceSynchronize());
            cpu = cpu_ms() - cpu;
            double wall =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            int info = 0;
            size_t written = 0;
            dss_check(cudssDataGet(context.handle, context.data, CUDSS_DATA_INFO, &info,
                                   sizeof(info), &written));
            require(info == 0, "cuDSS reorder failed");
            HostOrder host;
            host.p = context.array(CUDSS_DATA_PERM_REORDER_ROW);
            host.parts = context.array(CUDSS_DATA_ND_PARTITION_TREE);
            require(host.p == context.array(CUDSS_DATA_PERM_REORDER_COL),
                    "cuDSS row/column orders differ");
            require(host.p.size() == std::size_t(n), "cuDSS permutation length");
            host.inverse.resize(n, -1);
            for (int k = 0; k < n; ++k) {
                int u = host.p[k];
                require(u >= 0 && u < n && host.inverse[u] == -1, "invalid cuDSS permutation");
                host.inverse[u] = k;
            }
            host.levels = 0;
            while ((std::size_t(1) << host.levels) < host.parts.size() + 1)
                ++host.levels;
            validate(graph, host);
            if (rep >= 0) {
                if (host.p != previous) {
                    fill = exact_fill(graph, host.p);
                    previous = host.p;
                }
                std::printf("%s,%d,%zu,cudss_nd,%d,%.6f,%.6f,%llu,%d,0,0,0,0\n", name.c_str(), n,
                            graph.col.size(), rep, wall, cpu, (unsigned long long)fill,
                            host.levels);
                std::fflush(stdout);
            }
            if (rep == reps - 1 && argc > 4)
                save(std::string(argv[4]) + ".cudss", host);
        }
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
}

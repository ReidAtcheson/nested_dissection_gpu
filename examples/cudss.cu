#include "../tests/support.hpp"
#include "ndgpu/cudss.hpp"
#include <cmath>
#include <iostream>

static void dss_check(cudssStatus_t s) {
    if (s != CUDSS_STATUS_SUCCESS)
        throw std::runtime_error("cuDSS status " + std::to_string(int(s)));
}
// Small example-only owner for the cuDSS objects and double-valued matrix.
struct Solve {
    cudssHandle_t handle{};
    cudssConfig_t config{};
    cudssData_t data{};
    cudssMatrix_t matrix{}, b{}, x{};
    int *row = nullptr, *col = nullptr;
    double *values = nullptr, *rhs = nullptr, *solution = nullptr;
    ~Solve() {
        if (matrix)
            cudssMatrixDestroy(matrix);
        if (b)
            cudssMatrixDestroy(b);
        if (x)
            cudssMatrixDestroy(x);
        if (data)
            cudssDataDestroy(handle, data);
        if (config)
            cudssConfigDestroy(config);
        if (handle)
            cudssDestroy(handle);
        cudaFree(row);
        cudaFree(col);
        cudaFree(values);
        cudaFree(rhs);
        cudaFree(solution);
    }
};
int main(int argc, char **argv) try {
    if (argc > 2)
        throw std::invalid_argument("usage: ndgpu_cudss [grid-side=16]");
    cuda_check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    auto graph = grid(argc > 1 ? std::stoi(argv[1]) : 16);
    int n = graph.size();
    DeviceGraph device(graph);
    ndgpu::Ordering ordering(device.view);
    auto order = ordering.compute();
    HostOrder host(order);
    validate(graph, host);
    Solve s;
    dss_check(cudssCreate(&s.handle));
    dss_check(cudssConfigCreate(&s.config));
    dss_check(cudssDataCreate(s.handle, &s.data));
    // This is the entire ordering handoff; all three arrays already live on GPU.
    ndgpu::set_cudss_order(s.handle, s.config, s.data, order);
    std::vector<int> row(n + 1), col;
    std::vector<double> values;
    for (int u = 0; u < n; ++u) {
        for (int e = graph.row[u]; e < graph.row[u + 1]; ++e)
            if (graph.col[e] < u) {
                col.push_back(graph.col[e]);
                values.push_back(-1);
            }
        col.push_back(u);
        values.push_back(graph.row[u + 1] - graph.row[u] + 1);
        row[u + 1] = col.size();
    }
    // (Laplacian + I) * ones = ones: strictly SPD without altering the pattern.
    std::vector<double> ones(n, 1);
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&s.row), row.size() * sizeof(int)));
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&s.col), col.size() * sizeof(int)));
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&s.values), values.size() * sizeof(double)));
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&s.rhs), n * sizeof(double)));
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&s.solution), n * sizeof(double)));
    cuda_check(cudaMemcpy(s.row, row.data(), row.size() * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(s.col, col.data(), col.size() * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(s.values, values.data(), values.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(s.rhs, ones.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    dss_check(cudssMatrixCreateCsr(&s.matrix, n, n, col.size(), s.row, nullptr, s.col, s.values,
                                   CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_SPD,
                                   CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO));
    dss_check(cudssMatrixCreateDn(&s.b, n, 1, n, s.rhs, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    dss_check(cudssMatrixCreateDn(&s.x, n, 1, n, s.solution, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    for (auto phase : {CUDSS_PHASE_REORDERING, CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
                       CUDSS_PHASE_FACTORIZATION, CUDSS_PHASE_SOLVE}) {
        dss_check(cudssExecute(s.handle, phase, s.config, s.data, s.matrix, s.x, s.b));
        cuda_check(cudaDeviceSynchronize());
        int info = 0;
        size_t written = 0;
        dss_check(cudssDataGet(s.handle, s.data, CUDSS_DATA_INFO, &info, sizeof(info), &written));
        require(info == 0, "cuDSS numerical phase failed");
    }
    auto get_array = [&](cudssDataParam_t key) {
        size_t bytes = 0, written = 0;
        dss_check(cudssDataGet(s.handle, s.data, key, nullptr, 0, &bytes));
        std::vector<int> out(bytes / sizeof(int));
        dss_check(cudssDataGet(s.handle, s.data, key, out.data(), bytes, &written));
        require(bytes == written, "cuDSS array size mismatch");
        return out;
    };
    require(get_array(CUDSS_DATA_PERM_REORDER_ROW) == host.p, "cuDSS changed supplied permutation");
    require(get_array(CUDSS_DATA_ND_PARTITION_TREE) == host.parts, "cuDSS changed supplied tree");
    std::vector<double> solution(n);
    cuda_check(cudaMemcpy(solution.data(), s.solution, n * sizeof(double), cudaMemcpyDeviceToHost));
    double error = 0, residual = 0;
    for (int u = 0; u < n; ++u) {
        error += (solution[u] - 1) * (solution[u] - 1);
        double r = (graph.row[u + 1] - graph.row[u] + 1) * solution[u] - 1;
        for (int e = graph.row[u]; e < graph.row[u + 1]; ++e)
            r -= solution[graph.col[e]];
        residual += r * r;
    }
    error = std::sqrt(error / n);
    residual = std::sqrt(residual / n);
    require(std::isfinite(error) && std::isfinite(residual) && error < 1e-10 && residual < 1e-10,
            "cuDSS solve check failed");
    std::cout << "PASS: cuDSS permutation/tree round trip, relative error=" << error
              << ", residual=" << residual << '\n';
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}

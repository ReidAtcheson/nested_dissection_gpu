#include "../tests/support.hpp"
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>

static double cpu_ms() {
    timespec t{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec * 1000. + t.tv_nsec * 1e-6;
}
int main(int argc, char **argv) try {
    if (argc > 4)
        throw std::invalid_argument("usage: ndgpu_grid [side=32] [repetitions=3] [output-prefix]");
    int side = argc > 1 ? std::stoi(argv[1]) : 32, reps = argc > 2 ? std::stoi(argv[2]) : 3;
    if (reps < 1)
        throw std::invalid_argument("repetitions must be positive");
    cuda_check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    auto graph = grid(side);
    DeviceGraph device(graph);
    auto setup_start = std::chrono::steady_clock::now();
    ndgpu::Ordering ordering(device.view);
    std::fprintf(
        stderr, "workspace_setup_ms=%.3f\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setup_start)
            .count());
    ordering.compute(); // warmup
    std::puts("sample,n,wall_ms,cpu_ms,root_separator,leaves,max_depth,max_leaf,oversized_leaves,"
              "component_splits,peak_partitions");
    ndgpu::Order result{};
    for (int sample = 0; sample < reps; ++sample) {
        auto start = std::chrono::steady_clock::now();
        double cpu = cpu_ms();
        result = ordering.compute();
        cpu = cpu_ms() - cpu;
        double wall =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        auto stats = ordering.statistics();
        std::printf("%d,%d,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%d\n", sample, graph.size(), wall, cpu,
                    stats.root_separator, stats.leaves, stats.max_depth, stats.largest_leaf,
                    stats.oversized_leaves, stats.component_splits, stats.peak_batch);
    }
    HostOrder host(result);
    validate(graph, host);
    if (argc > 3)
        save(argv[3], host);
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}

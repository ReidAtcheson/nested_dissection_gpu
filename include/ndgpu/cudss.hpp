#pragma once

#include "ordering.hpp"
#include <cudss.h>
#include <stdexcept>
#include <string>

namespace ndgpu {
// Call before CUDSS_PHASE_REORDERING. Ordering must remain alive while cuDSS
// consumes these borrowed device arrays; compute() has already synchronized
// them.
inline void set_cudss_order(cudssHandle_t handle, cudssConfig_t config, cudssData_t data,
                            const Order &order) {
    auto check = [](cudssStatus_t status) {
        if (status != CUDSS_STATUS_SUCCESS)
            throw std::runtime_error("cuDSS order handoff: status " + std::to_string(int(status)));
    };
    auto algorithm = CUDSS_REORDERING_ALG_NESTED_DISSECTION;
    check(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &algorithm, sizeof(algorithm)));
    check(cudssConfigSet(config, CUDSS_CONFIG_ND_NLEVELS, &order.num_levels,
                         sizeof(order.num_levels)));
    check(cudssDataSet(handle, data, CUDSS_DATA_USER_PERM, order.permutation,
                       std::size_t(order.n) * sizeof(std::int32_t)));
    check(cudssDataSet(handle, data, CUDSS_DATA_USER_ND_PARTITION_TREE, order.parts,
                       std::size_t(order.num_parts) * sizeof(std::int32_t)));
}
} // namespace ndgpu

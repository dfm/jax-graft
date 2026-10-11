// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef METAL_PJRT_COMPILER_PASSES_HOST_COMPUTE_LAYOUT_H_
#define METAL_PJRT_COMPILER_PASSES_HOST_COMPUTE_LAYOUT_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Gives each host-offloaded region (compute_on("device_host"): a host-thread
// async-start wrapping a HostExecute custom call) its operands' layouts as
// parameter layouts. GPU layout assignment runs on the main thread only: it
// lays out the async-start's operands but leaves the wrapped computation's
// parameters, which the HLO verifier then rejects (JAX's gradient of a
// LAPACK call in a region: a column-major f64[n,1] residual passed between
// regions). Where the HostExecute's computation (the CPU code) expects
// another layout, a copy at its start converts it on the CPU. Runs right
// after layout assignment, on every execution thread.
class MetalHostComputeLayoutFixer : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "metal-host-compute-layout-fixer";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // METAL_PJRT_COMPILER_PASSES_HOST_COMPUTE_LAYOUT_H_

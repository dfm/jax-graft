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

// Gives each host-offloaded region (jax.experimental.compute_on
// ("device_host"): an async-start on the host thread wrapping a HostExecute
// custom call) the parameter layouts its operands have.
//
// GPU layout assignment runs on the main thread only. It lays out the
// async-start's operands (e.g. a MoveToDevice of another region's result
// becomes row-major, with a copy) but leaves the wrapped computation's
// parameters as they were, so the HLO verifier rejects the module (async
// shape does not match the async computation's parameters). JAX's gradient
// of a LAPACK call in a region does this: a column-major f64[n,1] residual
// of the forward region feeds the backward one. The HostExecute's own
// computation (the CPU code) already takes the operand's layout in that
// case; where it does not, a copy at its start converts the operand to the
// layout it expects, on the CPU.
//
// Runs right after layout assignment, on every execution thread.
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

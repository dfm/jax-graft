// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/host_compute_layout.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace gpu {
namespace {

// Makes `param` (a parameter of a HostExecute's computation) take `layout`'s
// layout; its users keep seeing the old one, through a copy.
absl::Status RelayoutParameter(HloInstruction* param, const Shape& layout) {
  if (Shape::Equal()(param->shape(), layout)) return absl::OkStatus();
  const Shape old = param->shape();
  std::vector<HloInstruction*> users = param->users();
  const bool was_root = param->parent()->root_instruction() == param;
  *param->mutable_shape() = layout;
  HloInstruction* copy = param->parent()->AddInstruction(
      HloInstruction::CreateUnary(old, HloOpcode::kCopy, param));
  for (HloInstruction* user : users) {
    TF_RETURN_IF_ERROR(param->ReplaceUseWith(user, copy));
  }
  if (was_root) param->parent()->set_root_instruction(copy);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> MetalHostComputeLayoutFixer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp : module->computations(execution_threads)) {
    for (HloInstruction* start : comp->instructions()) {
      if (start->opcode() != HloOpcode::kAsyncStart ||
          start->async_execution_thread() != HloInstruction::kHostThread) {
        continue;
      }
      HloComputation* wrapped = start->async_wrapped_computation();
      for (int64_t i = 0; i < start->operand_count(); ++i) {
        const Shape& want = start->operand(i)->shape();
        HloInstruction* param = wrapped->parameter_instruction(i);
        if (Shape::Equal()(param->shape(), want)) continue;
        if (!ShapeUtil::Compatible(param->shape(), want)) {
          return absl::InternalError(absl::StrCat(
              "Metal: host-offloaded region ", start->name(), " operand ", i,
              " has shape ", want.ToString(), " but its computation takes ",
              param->shape().ToString()));
        }
        *param->mutable_shape() = want;
        changed = true;
        for (HloInstruction* user : param->users()) {
          if (!user->IsCustomCall("HostExecute")) continue;
          HloComputation* body = user->called_computations().front();
          for (int64_t k : user->OperandIndices(param)) {
            TF_RETURN_IF_ERROR(
                RelayoutParameter(body->parameter_instruction(k), want));
          }
        }
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla

// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/host_compute_layout.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/layout_util.h"
#include "xla/service/hlo_verifier.h"

namespace xla {
namespace gpu {
namespace {

using MetalHostComputeLayoutFixerTest = HloHardwareIndependentTestBase;

// Layout-sensitive, as at the end of XLA:GPU's post-layout pipeline.
bool Verifies(HloModule* module) {
  return HloVerifier(/*layout_sensitive=*/true, /*allow_mixed_precision=*/true)
      .Run(module)
      .ok();
}

// What GPU layout assignment leaves for the gradient of a LAPACK call in a
// compute_on("device_host") region: the async-start's operand is row-major,
// the wrapped computation's parameter still column-major. $B is the layout
// the HostExecute's computation (the CPU code) takes.
constexpr char kModule[] = R"(
HloModule m

body {
  p0 = f64[8,4]$B parameter(0)
  ROOT n = f64[8,4]$B negate(p0)
}

wrapped {
  w0 = f64[8,4]{0,1} parameter(0)
  ROOT h = f64[8,4]$B custom-call(w0), custom_call_target="HostExecute", called_computations={body}
}, execution_thread="host"

ENTRY e {
  x = f64[8,4]{1,0} parameter(0)
  s = ((f64[8,4]{1,0}), f64[8,4]$B, u32[]) async-start(x), async_execution_thread="host", calls=wrapped
  ROOT d = f64[8,4]$B async-done(s)
})";

TEST_F(MetalHostComputeLayoutFixerTest, CopiesWhereTheCpuCodeWantsAnother) {
  auto module = ParseAndReturnUnverifiedModule(
      absl::StrReplaceAll(kModule, {{"$B", "{0,1}"}}));
  ASSERT_TRUE(module.ok()) << module.status();
  // The verifier's complaint (async shape vs the computation's parameters).
  EXPECT_FALSE(Verifies(module->get()));
  MetalHostComputeLayoutFixer pass;
  auto changed = RunHloPass(&pass, module->get());
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_TRUE(*changed);
  HloModule* m = module->get();
  const HloInstruction* w0 =
      m->GetComputationWithName("wrapped")->parameter_instruction(0);
  EXPECT_EQ(w0->shape().layout(), LayoutUtil::MakeLayout({1, 0}));
  // The CPU code takes the row-major buffer and copies it to column-major.
  const HloComputation* body = m->GetComputationWithName("body");
  EXPECT_EQ(body->parameter_instruction(0)->shape().layout(),
            LayoutUtil::MakeLayout({1, 0}));
  const HloInstruction* neg = body->root_instruction();
  ASSERT_EQ(neg->operand(0)->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(neg->operand(0)->shape().layout(), LayoutUtil::MakeLayout({0, 1}));
  EXPECT_TRUE(Verifies(m));
}

TEST_F(MetalHostComputeLayoutFixerTest, NoCopyWhereTheCpuCodeAgrees) {
  auto module = ParseAndReturnUnverifiedModule(
      absl::StrReplaceAll(kModule, {{"$B", "{1,0}"}}));
  ASSERT_TRUE(module.ok()) << module.status();
  MetalHostComputeLayoutFixer pass;
  auto changed = RunHloPass(&pass, module->get());
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_TRUE(*changed);
  const HloComputation* body = (*module)->GetComputationWithName("body");
  EXPECT_EQ(body->root_instruction()->operand(0)->opcode(),
            HloOpcode::kParameter);
  EXPECT_TRUE(Verifies(module->get()));
  // Nothing left to do.
  changed = RunHloPass(&pass, module->get());
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_FALSE(*changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

#include "metal_pjrt/compiler/passes/hlo_checks.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "metal_pjrt/blas/blas_lt_support.h"
#include "metal_pjrt/compiler/passes/dot_upcast.h"
#include "metal_pjrt/compiler/report_bug.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/primitive_util.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

std::string DescribeOp(const HloInstruction& instr) {
  const OpMetadata& md = instr.metadata();
  std::string s(instr.name());
  if (!md.op_name().empty()) absl::StrAppend(&s, " (", md.op_name(), ")");
  const HloModule* module = instr.GetModule();
  if (md.stack_frame_id() != 0 && module != nullptr) {
    HloStackFrame f =
        module->get_stack_frame(StackFrameId{md.stack_frame_id()});
    if (!f.empty()) {
      absl::StrAppend(&s, " at ", f.file_name, ":", f.line);
      return s;
    }
  }
  if (!md.source_file().empty()) {
    absl::StrAppend(&s, " at ", md.source_file(), ":", md.source_line());
  }
  return s;
}

namespace {

// XLA's GetBatchRowColumnShape (matmul_utils.cc, under GemmConfig::For)
// multiplies each group of dimensions (batch, rows, columns) with an int
// accumulator (absl::c_accumulate(dims, 1, ...)), so a group of 2^31 or more
// elements wraps and the thunk would run the GEMM on the wrapped size. Checks
// the HLO's int64 dimensions before that.
absl::Status CheckGemmGroupsFitInt32(const HloInstruction& instr,
                                     const DotDimensionNumbers& dnums) {
  auto check = [](const Shape& shape, absl::Span<const int64_t> batch,
                  absl::Span<const int64_t> contracting) -> absl::Status {
    int64_t b = 1, c = 1, rest = 1;
    for (int64_t d = 0; d < shape.dimensions().size(); ++d) {
      const int64_t size = shape.dimensions(d);
      if (absl::c_linear_search(batch, d)) {
        b *= size;
      } else if (absl::c_linear_search(contracting, d)) {
        c *= size;
      } else {
        rest *= size;
      }
    }
    for (int64_t group : {b, c, rest}) {
      if (group > std::numeric_limits<int32_t>::max()) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: matmul operand ", ShapeUtil::HumanString(shape),
            " has a batch, row or column group of ", group,
            " elements; XLA's matmul config counts them in 32 bits. Split "
            "the batch (lax.map, or chunks)"));
      }
    }
    return absl::OkStatus();
  };
  TF_RETURN_IF_ERROR(check(instr.operand(0)->shape(),
                           dnums.lhs_batch_dimensions(),
                           dnums.lhs_contracting_dimensions()));
  return check(instr.operand(1)->shape(), dnums.rhs_batch_dimensions(),
               dnums.rhs_contracting_dimensions());
}

absl::Status CheckGemm(const HloInstruction& instr) {
  if (!IsCublasLtMatmul(instr)) {
    return absl::UnimplementedError(
        absl::StrCat("Metal: no matmul implementation for ",
                     instr.custom_call_target(), " (", DescribeOp(instr), ")"));
  }
  TF_ASSIGN_OR_RETURN(auto config, instr.backend_config<GpuBackendConfig>());
  // What GetMatmulPlan checks again at run time, on the config the thunk
  // builds.
  absl::Status s = [&]() -> absl::Status {
    TF_RETURN_IF_ERROR(CheckGemmGroupsFitInt32(
        instr, config.gemm_backend_config().dot_dimension_numbers()));
    TF_ASSIGN_OR_RETURN(
        stream_executor::gpu::BlasLt::Epilogue e,
        gpublas_lt::AsBlasLtEpilogue(config.gemm_backend_config().epilogue()));
    TF_ASSIGN_OR_RETURN(
        GemmConfig gemm,
        GemmConfig::For(&instr,
                        se::GpuComputeCapability(se::OneAPIComputeCapability())));
    TF_ASSIGN_OR_RETURN(stream_executor::metal::ValidatedMatmul v,
                        stream_executor::metal::ValidateMatmul(gemm, e));
    // GemmRewriter's FuseVectorBiasAdd looks through a slice or a bitcast
    // of the matmul and compares the bias with that shape's minor
    // dimension, so (x @ w)[:, :3] + b and (x @ w).reshape(m, h, d) + b
    // arrive with fewer bias elements than the matmul has columns. The
    // kernels index the bias by column.
    if (v.epi.bias) {
      const int bias_index =
          config.gemm_backend_config().beta() != 0.0 ? 3 : 2;
      if (instr.operand_count() > bias_index &&
          ShapeUtil::ElementsIn(instr.operand(bias_index)->shape()) != v.n) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: XLA fused a bias of ",
            ShapeUtil::ElementsIn(instr.operand(bias_index)->shape()),
            " elements into a matmul with ", v.n,
            " output columns, through a slice or reshape of the matmul's "
            "result. Slice or reshape the matmul's operands instead, or add "
            "a bias as wide as the matmul before slicing or reshaping"));
      }
    }
    return absl::OkStatus();
  }();
  if (s.ok()) return s;
  return absl::Status(s.code(),
                      absl::StrCat(s.message(), " (", DescribeOp(instr), ")"));
}

// Ops that compute on element values (as opposed to moving them, calling
// something, or running on the host).
bool Computes(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kSelect ||
      instr.opcode() == HloOpcode::kConstant) {
    return false;
  }
  if (instr.IsElementwise()) return true;
  switch (instr.opcode()) {
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kDot:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kConvolution:
    case HloOpcode::kFft:
    case HloOpcode::kCholesky:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kSort:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kRng:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kMap:
    case HloOpcode::kTopK:
      return true;
    default:
      return false;
  }
}

bool TouchesType(const HloInstruction& instr,
                 absl::Span<const PrimitiveType> types) {
  auto has = [&](const Shape& shape) {
    for (PrimitiveType t : types) {
      if (ShapeUtil::HasPrimitiveType(shape, t)) return true;
    }
    return false;
  };
  if (has(instr.shape())) return true;
  for (const HloInstruction* op : instr.operands()) {
    if (has(op->shape())) return true;
  }
  return false;
}

// Ops that move f64 values through an emitted kernel, which cannot load f64
// (the MSL emitter would refuse it without naming the op). Contiguous
// slices, dynamic slices and updates, reshapes and bitcasts are copies or
// no-ops and work.
bool MovesF64InAKernel(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kTranspose:
      if (ShapeUtil::TransposeIsBitcast(instr.operand(0)->shape(),
                                        instr.shape(), instr.dimensions())) {
        return false;
      }
      [[fallthrough]];
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kGather:
    case HloOpcode::kIota:
    case HloOpcode::kPad:
    case HloOpcode::kReverse:
    case HloOpcode::kSelect:
      return TouchesType(instr, {F64, C128});
    default:
      return false;
  }
}

// A scatter whose combiner does more than overwrite needs atomics on the
// element type (read-modify-write when indices collide). For complex64,
// MetalComplexScatterSplitter has made adds into two f32 scatters, and the
// kernel lowering stores a single-operand overwrite with one 8-byte store
// (metal_kernel_compiler.cc) where XLA would compare-and-swap; any other
// complex combiner would need a 64-bit compare-and-swap.
bool NeedsWideAtomics(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kScatter ||
      Cast<HloScatterInstruction>(&instr)->unique_indices()) {
    return false;
  }
  const HloInstruction* root = instr.to_apply()->root_instruction();
  const int n = instr.operand_count() / 2;  // operands, indices, updates
  bool overwrite = root->opcode() == HloOpcode::kParameter &&
                   root->parameter_number() >= n;
  if (root->opcode() == HloOpcode::kTuple) {
    overwrite = true;
    for (const HloInstruction* e : root->operands()) {
      overwrite &= e->opcode() == HloOpcode::kParameter &&
                   e->parameter_number() >= n;
    }
  }
  if (TouchesType(instr, {C64, C128})) {
    return !overwrite || instr.operand_count() != 3;
  }
  if (overwrite) return false;
  bool wide = false;
  ShapeUtil::ForEachSubshape(instr.shape(), [&](const Shape& s,
                                                const ShapeIndex&) {
    if (s.IsArray() && primitive_util::BitWidth(s.element_type()) > 32) {
      wide = true;
    }
  });
  return wide;
}

// Whether XLA:CPU can compile this custom call in a host-offloaded region:
// one of its own targets (cpu/thunk_emitter.cc), or a handler registered for
// the host in this binary (cpu/runtime/custom_call_thunk.cc looks there).
// JAX lowers compute_on("device_host") regions with its CPU rules, whose
// LAPACK calls (lapack_*_ffi) jaxlib registers in its own copy of XLA, not
// in this one.
bool HostCanRun(const HloInstruction& instr) {
  const std::string& target = instr.custom_call_target();
  if (target == "TopK" || target == "SliceToDynamic" ||
      target == "GetRngSeed") {
    return true;
  }
  // Annotations JAX may emit in a region, which XLA removes on the main
  // thread before it moves the region to the host (ShardingRemover,
  // ConvertMemoryPlacementToInternalAnnotations).
  if (target == "Sharding" || target == "SPMDFullToShardShape" ||
      target == "SPMDShardToFullShape" ||
      target == "annotate_device_placement") {
    return true;
  }
  if (Cast<HloCustomCallInstruction>(&instr)->api_version() ==
      CustomCallApiVersion::API_VERSION_TYPED_FFI) {
    return ffi::FindHandler(target, "Host").ok();
  }
  return CustomCallTargetRegistry::Global()->Lookup(target, "Host") != nullptr;
}

}  // namespace

absl::Status WithHostComputeHidden(HloModule* module,
                                   absl::FunctionRef<absl::Status()> fn) {
  // One caller per computation, as XLA's pre-SPMD FlattenCallGraph leaves it
  // later, but before HloHostDeviceTypeCallWrapper: that pass marks every op
  // of a region host-typed, then skips a computation only if its single
  // caller is host-typed. A reducer, comparator or helper called twice in a
  // region (two sorts, jax.random twice) would have its marked ops offloaded
  // again, as an async region nested in the region, which XLA:CPU cannot
  // compile (a process abort).
  if (absl::c_any_of(
          module->computations({HloInstruction::kMainExecutionThread}),
          [](const HloComputation* comp) {
            return absl::c_any_of(
                comp->instructions(), [](const HloInstruction* instr) {
                  auto it = instr->frontend_attributes().map().find(
                      "_xla_compute_type");
                  return it != instr->frontend_attributes().map().end() &&
                         it->second == "host";
                });
          })) {
    TF_RETURN_IF_ERROR(FlattenCallGraph().Run(module).status());
  }
  std::vector<HloComputation*> hidden;
  absl::flat_hash_set<HloComputation*> seen;
  for (HloComputation* comp :
       module->computations({HloInstruction::kMainExecutionThread})) {
    for (HloInstruction* instr : comp->instructions()) {
      // kXlaComputeTypeAttr / kXlaComputeTypeHost (xla/side_effect_util.h).
      const auto& attributes = instr->frontend_attributes().map();
      auto type = attributes.find("_xla_compute_type");
      if (type == attributes.end() || type->second != "host") continue;
      for (HloComputation* called : instr->called_computations()) {
        if (seen.insert(called).second) hidden.push_back(called);
      }
    }
  }
  for (size_t i = 0; i < hidden.size(); ++i) {
    for (HloInstruction* instr : hidden[i]->instructions()) {
      for (HloComputation* called : instr->called_computations()) {
        if (seen.insert(called).second) hidden.push_back(called);
      }
    }
  }
  // A host-typed op that is not a call (the legacy `with compute_on(...)` of
  // JAX 0.10 marks each op) cannot be hidden, and the plugin's passes could
  // rewrite it into a metal$ call that XLA:CPU then cannot compile.
  for (HloComputation* comp : module->computations()) {
    if (seen.contains(comp)) continue;
    for (HloInstruction* instr : comp->instructions()) {
      const auto& attributes = instr->frontend_attributes().map();
      auto type = attributes.find("_xla_compute_type");
      if (type == attributes.end() || type->second != "host") continue;
      // The region's call, and the results taken from it (the HLO import
      // copies the call's attributes onto them).
      if (instr->opcode() == HloOpcode::kCall ||
          (instr->opcode() == HloOpcode::kGetTupleElement &&
           instr->operand(0)->opcode() == HloOpcode::kCall)) {
        continue;
      }
      return absl::UnimplementedError(absl::StrCat(
          "Metal: this form of compute_on(\"device_host\") is not "
          "supported (it marks single ops); use "
          "jax.experimental.compute_on.compute_on(f, "
          "compute_type=\"device_host\", out_memory_spaces=...) "
          "(JAX 0.11 or later): ",
          DescribeOp(*instr)));
    }
  }
  // XLA CHECK-fails (aborts the process) when XLA:CPU cannot compile a host
  // region, e.g. on a custom call it has no handler for.
  for (HloComputation* comp : hidden) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kCustomCall && !HostCanRun(*instr)) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: ", instr->custom_call_target(),
            " inside jax.experimental.compute_on(\"device_host\") is not "
            "supported (JAX lowers that region for the CPU, and this plugin "
            "has no CPU implementation of it); move it out of the region: ",
            DescribeOp(*instr)));
      }
    }
  }
  // JAX marks every op of a region host-typed, and XLA would offload each
  // marked op inside it again (HloHostDeviceTypeCallWrapper,
  // HostComputeAsyncifier), as an async region nested in the region, which
  // XLA:CPU cannot compile (a process abort). The region's own call is what
  // carries the offload; the marks inside it are dropped.
  for (HloComputation* comp : hidden) {
    for (HloInstruction* instr : comp->instructions()) {
      instr->erase_frontend_attribute("_xla_compute_type");
    }
  }
  // While hidden, a main-thread call reaches a host-thread computation,
  // which HloVerifier rejects; `fn` must not run it.
  for (HloComputation* comp : hidden) {
    comp->SetExecutionThread(HloInstruction::kHostThread);
  }
  absl::Status status = fn();
  for (HloComputation* comp : hidden) {
    comp->SetExecutionThread(HloInstruction::kMainExecutionThread);
  }
  return status;
}

absl::Status CheckPostGemmRewriter(const HloModule& module) {
  // An async op inside a host-offloaded region (a region nested in it)
  // would make XLA:CPU's compile fail, which aborts the process.
  for (const HloComputation* comp : module.computations()) {
    if (comp->execution_thread() != HloInstruction::kHostThread) continue;
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kAsyncStart ||
          instr->opcode() == HloOpcode::kAsyncUpdate ||
          instr->opcode() == HloOpcode::kAsyncDone) {
        return absl::InternalError(absl::StrCat(
            "Metal: an async op inside a compute_on(\"device_host\") "
            "region: ",
            DescribeOp(*instr), metal_pjrt::kReportBug));
      }
    }
  }
  // Host-offloaded computations run on the CPU (see WithHostComputeHidden).
  for (const HloComputation* comp :
       module.computations({HloInstruction::kMainExecutionThread})) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() == HloOpcode::kRng ||
          instr->opcode() == HloOpcode::kRngGetAndUpdateState) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: the HLO rng op (jax.lax.rng_uniform) is not supported "
            "(XLA generates it with a kernel that has no Metal translation); "
            "use jax.random: ",
            DescribeOp(*instr)));
      }
      if (instr->opcode() == HloOpcode::kMulhi &&
          primitive_util::BitWidth(instr->shape().element_type()) > 32) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: jax.lax.mulhi on 64-bit integers is not supported (XLA "
            "computes it in 128-bit integers, which Metal does not have): ",
            DescribeOp(*instr)));
      }
      if (instr->opcode() == HloOpcode::kFft) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: the HLO fft op is not supported (XLA's FFT runs on cuFFT "
            "only); jax.numpy.fft / jax.lax.fft lower to the plugin's "
            "metal$fft instead, so this program was lowered for another "
            "platform: ",
            DescribeOp(*instr)));
      }
      if (Computes(*instr) && TouchesType(*instr, {F64, C128})) {
        return absl::UnimplementedError(absl::StrCat(
            "Metal: f64 / complex128 arithmetic is not supported (Apple GPUs "
            "have no double precision; use float32 / complex64, or run it on "
            "the CPU backend): ",
            DescribeOp(*instr)));
      }
      if (MovesF64InAKernel(*instr)) {
        const bool c128 = TouchesType(*instr, {C128});
        return absl::UnimplementedError(absl::StrCat(
            "Metal: ", c128 ? "complex128 " : "f64 ",
            HloOpcodeString(instr->opcode()),
            " inside jit is not supported (Apple GPUs have no double "
            "precision, so GPU kernels cannot load f64; only transfers of "
            "f64 / complex128 arrays work). Use float32 / complex64, or run "
            "it on the CPU backend: ",
            DescribeOp(*instr)));
      }
      if (NeedsWideAtomics(*instr)) {
        if (TouchesType(*instr, {C64, C128})) {
          return absl::UnimplementedError(absl::StrCat(
              "Metal: a scatter of complex values with this combiner needs "
              "64-bit atomics, which Metal does not have; pass "
              "unique_indices=True if the indices do not repeat (scatters "
              "that add the update or overwrite with it do not need it): ",
              DescribeOp(*instr)));
        }
        return absl::UnimplementedError(absl::StrCat(
            "Metal: scatter with a combiner on 64-bit elements needs 64-bit "
            "atomics, which Metal does not have: ",
            DescribeOp(*instr)));
      }
      if (instr->opcode() == HloOpcode::kConvolution &&
          instr->batch_group_count() > 1) {
        return absl::InternalError(absl::StrCat(
            "Metal: convolution with batch_group_count > 1 after "
            "ConvolutionGroupConverter (XLA's loop emitter computes it "
            "wrong): ",
            DescribeOp(*instr), metal_pjrt::kReportBug));
      }
      if (instr->opcode() == HloOpcode::kDot &&
          instr->shape().element_type() == C64) {
        return absl::InternalError(absl::StrCat(
            "Metal: complex64 dot after MetalComplexDotExpander: ",
            DescribeOp(*instr), metal_pjrt::kReportBug));
      }
      if (IsNarrowOperandDot(instr)) {
        return absl::InternalError(absl::StrCat(
            "Metal: dot with operands narrower than its result after "
            "MetalDotOperandUpcaster: ",
            DescribeOp(*instr), metal_pjrt::kReportBug));
      }
      if (instr->opcode() != HloOpcode::kCustomCall) continue;
      if (IsCublasGemm(*instr)) TF_RETURN_IF_ERROR(CheckGemm(*instr));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

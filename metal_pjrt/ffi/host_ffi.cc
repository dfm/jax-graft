// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// metal_pjrt_register_host_ffi_handler: lets the frontend register jaxlib's
// CPU FFI handlers (LAPACK, ...) with this library's copy of XLA, for the
// host platform.
//
// jax.experimental.compute_on("device_host") regions run on the CPU: XLA
// compiles them with XLA:CPU inside this library (HostExecute), and JAX
// lowers them with its CPU rules, so they call jaxlib's CPU custom calls
// (lapack_dpotrf_ffi, ...). jaxlib registers those with its own copy of
// XLA, not this one. The XLA FFI is a C ABI meant to cross such copies
// (XLA checks the handler's FFI API version here), so the frontend hands
// over the same handler pointers (jax_graft/_host_ffi.py).
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"

// Returns 0 when registered, 1 when this library already has a handler
// under that name for the host (its own, which is kept), and -1 on failure
// with a NUL-terminated message in `error` (at most `error_capacity` bytes).
// The stage pointers are XLA_FFI_Handler*, null for absent stages.
extern "C" __attribute__((visibility("default"))) int
metal_pjrt_register_host_ffi_handler(const char* name, void* instantiate,
                                     void* prepare, void* initialize,
                                     void* execute, void* record, char* error,
                                     int64_t error_capacity) {
  if (xla::ffi::FindHandler(name, "Host").ok()) return 1;
  XLA_FFI_Handler_Bundle bundle = {
      reinterpret_cast<XLA_FFI_Handler*>(instantiate),
      reinterpret_cast<XLA_FFI_Handler*>(prepare),
      reinterpret_cast<XLA_FFI_Handler*>(initialize),
      reinterpret_cast<XLA_FFI_Handler*>(execute),
      reinterpret_cast<XLA_FFI_Handler*>(record)};
  absl::Status status = xla::ffi::RegisterHandler(
      xla::ffi::GetXlaFfiApi(), name, "Host", bundle, /*traits=*/0);
  if (status.ok()) return 0;
  if (error != nullptr && error_capacity > 0) {
    std::string message(status.message());
    size_t n = std::min<size_t>(message.size(), error_capacity - 1);
    std::memcpy(error, message.data(), n);
    error[n] = '\0';
  }
  return -1;
}

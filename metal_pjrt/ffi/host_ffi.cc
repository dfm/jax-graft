// Copyright 2026 The jax-graft Authors
// SPDX-License-Identifier: Apache-2.0

// metal_pjrt_register_host_ffi_handler: registers one of jaxlib's CPU FFI
// handlers (its stateless linear algebra) with this library's copy of XLA,
// for the host. JAX lowers jax.experimental.compute_on("device_host")
// regions with its CPU rules, so they call jaxlib's custom calls
// (lapack_dpotrf_ffi, ...), which jaxlib registers with its own copy of XLA.
// The XLA FFI is a C ABI meant to cross such copies (RegisterHandler checks
// the handler's FFI API version). Caller: jax_graft/_host_ffi.py.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"

// `execute` is an XLA_FFI_Handler*. Returns 0, or -1 with a NUL-terminated
// message in `error` (at most `error_capacity` bytes).
extern "C" __attribute__((visibility("default"))) int
metal_pjrt_register_host_ffi_handler(const char* name, void* execute,
                                     char* error, int64_t error_capacity) {
  XLA_FFI_Handler_Bundle bundle = {};
  bundle.execute = reinterpret_cast<XLA_FFI_Handler*>(execute);
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

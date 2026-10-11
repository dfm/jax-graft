# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""jaxlib's CPU linear algebra in ``compute_on("device_host")`` regions.

JAX lowers such a region with its CPU rules, so a Cholesky in it calls
jaxlib's ``lapack_dpotrf_ffi``, which jaxlib registers with its own copy of
XLA, not the library's. The XLA FFI is a C ABI meant to cross copies of XLA,
so the first time a region is lowered for mtl this loads LAPACK
(``initialize()``) and hands jaxlib's linear algebra handlers (all stateless)
to the library (``metal_pjrt_register_host_ffi_handler``).

Nothing else is forwarded: XLA runs a region without an execution context,
where handlers with per-execution user data or state (jaxlib's CPU
callbacks, a user's own "cpu" FFI functions) would fail or crash; they are
refused at compile time. Python callbacks use the plugin's own handler
(_callbacks.py). A deserialized executable that was never lowered in this
process finds no handler and fails to load.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_registered = False
_register = None

_capsule_name = ctypes.pythonapi.PyCapsule_GetName
_capsule_name.restype = ctypes.c_char_p
_capsule_name.argtypes = [ctypes.py_object]
_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _register_linalg() -> None:
  """Load LAPACK and hand jaxlib's linear algebra handlers to the library,
  once."""
  global _registered
  with _lock:
    if _registered:
      return
    _registered = True
    from jax._src.lib import lapack
    lapack._lapack.initialize()
    for name, capsule, api_version in lapack.registrations()["cpu"]:
      if api_version != 1:  # not a typed-FFI handler
        continue
      execute = _capsule_pointer(capsule, _capsule_name(capsule))
      error = ctypes.create_string_buffer(1024)
      if _register(name.encode(), execute, error, len(error)) < 0:
        logger.warning(
            "jax-graft: cannot use jaxlib's %s in compute_on(\"device_host\") "
            "regions: %s", name, error.value.decode(errors="replace"))


def install(library_path) -> None:
  """Register jaxlib's linear algebra handlers when a compute_on region is
  first lowered for mtl. Call after the plugin is registered."""
  global _register
  if _register is not None:
    return
  lib = ctypes.CDLL(os.fspath(library_path))
  fn = lib.metal_pjrt_register_host_ffi_handler
  fn.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p,
                 ctypes.c_int64]
  fn.restype = ctypes.c_int
  _register = fn

  from jax._src import compute_on
  from jax._src.interpreters import mlir
  from jax_graft import PLATFORM
  upstream = mlir._lowerings[compute_on.compute_on_p]

  def lowering(ctx, *args, **params):
    try:
      _register_linalg()
    except Exception as e:  # noqa: BLE001 - the region then fails to compile
      logger.warning("jax-graft: jaxlib's linear algebra is unavailable in "
                     "compute_on(\"device_host\") regions: %s", e)
    return upstream.rule(ctx, *args, **params)
  mlir.register_lowering(compute_on.compute_on_p, lowering, platform=PLATFORM,
                         inline=upstream.inline)

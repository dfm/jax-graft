# Copyright 2026 The jax-graft Authors
# SPDX-License-Identifier: Apache-2.0

"""jaxlib's CPU FFI handlers in ``compute_on("device_host")`` regions.

Such a region runs on the CPU: XLA compiles it with XLA:CPU inside the
plugin's library, and JAX lowers it with its CPU rules, so it calls jaxlib's
CPU custom calls (``lapack_dpotrf_ffi`` for a Cholesky, ...). jaxlib
registers their handlers with its own copy of XLA, which the plugin's copy
cannot see. The XLA FFI is a C ABI meant to cross copies of XLA, so before
each compile this hands every typed-FFI handler jaxlib has for the host
(LAPACK, and any a user registered with ``jax.ffi.register_ffi_target`` for
"cpu") to the library (``metal_pjrt_register_host_ffi_handler``,
metal_pjrt/ffi/host_ffi.cc). A handler the library cannot take (e.g. built
against an incompatible FFI API version) is logged, and a region calling it
fails to compile with an error naming it.

Python callbacks in a region are not jaxlib's: they lower to the plugin's
own ``xla_ffi_python_metal_callback`` (_callbacks.py), which the library
also registers for the host.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_forwarded: set[str] = set()
_register = None
_STAGES = ("instantiate", "prepare", "initialize", "execute", "record")

_capsule_name = ctypes.pythonapi.PyCapsule_GetName
_capsule_name.restype = ctypes.c_char_p
_capsule_name.argtypes = [ctypes.py_object]
_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _pointer(capsule) -> int:
  return _capsule_pointer(capsule, _capsule_name(capsule))


def _host_targets():
  """(name, {stage: capsule}) for jaxlib's typed-FFI host handlers."""
  from jaxlib import xla_client
  # Registered with jaxlib's XLA (once its CPU backend is up)...
  for name, stages in xla_client._xla.custom_call_targets("Host").items():
    yield name, stages
  # ... or still queued for it (JAX_PLATFORMS without cpu).
  for name, fn, api_version, _ in list(
      xla_client._custom_callback.get("Host", ())):
    if api_version == 1:
      yield name, fn if isinstance(fn, dict) else {"execute": fn}


def sync() -> None:
  """Hand the library the host handlers it has not been given yet."""
  if _register is None:
    return
  with _lock:
    for name, stages in _host_targets():
      if name in _forwarded:
        continue
      _forwarded.add(name)
      if not isinstance(stages, dict) or "execute" not in stages:
        continue
      try:
        ptrs = [_pointer(stages[s]) if s in stages else None for s in _STAGES]
      except Exception as e:  # not an FFI handler capsule
        logger.info("not forwarding host FFI target %s: %s", name, e)
        continue
      error = ctypes.create_string_buffer(1024)
      rc = _register(name.encode(), *ptrs, error, len(error))
      if rc < 0:
        logger.warning(
            "jax_graft: cannot use jaxlib's host FFI handler %s in "
            "compute_on(\"device_host\") regions: %s", name,
            error.value.decode(errors="replace"))


def install(library_path) -> None:
  """Forward host handlers before every compile (and persistent-cache load:
  a cached host executable looks its handlers up when it is loaded). Call
  after the plugin is registered."""
  global _register
  if _register is not None:
    return
  lib = ctypes.CDLL(os.fspath(library_path))
  fn = lib.metal_pjrt_register_host_ffi_handler
  fn.argtypes = [ctypes.c_char_p] + [ctypes.c_void_p] * 5 + [
      ctypes.c_char_p, ctypes.c_int64]
  fn.restype = ctypes.c_int
  _register = fn

  from jax._src import compiler
  upstream = compiler.compile_or_get_cached

  def compile_or_get_cached(*args, **kwargs):
    sync()
    return upstream(*args, **kwargs)
  compile_or_get_cached.__wrapped__ = upstream
  compiler.compile_or_get_cached = compile_or_get_cached

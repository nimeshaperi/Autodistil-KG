"""
Compatibility shims for NVIDIA libraries and torch.compile.

This module MUST be imported before any other imports in main.py because:

1. Unsloth and torch.compile environment variables must be set before torch loads.
2. NVIDIA shared libraries (e.g. libcusparseLt) installed via pip need to be
   preloaded with ctypes because setting LD_LIBRARY_PATH in Python is too late
   for the dynamic linker.
3. The torch.compile inductor wrapper sometimes passes unknown options that
   cause RuntimeError; we patch it to silently filter unknown keys.
"""
import ctypes
import os
import shutil
import sys
from pathlib import Path

# Disable unsloth/torch.compile before any other imports
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Preload NVIDIA pip-installed shared libs
for _nvidia_base in [
    Path(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia"),
    Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia",
]:
    if _nvidia_base.exists():
        for _so in sorted(_nvidia_base.glob("*/lib/*.so*")):
            try:
                ctypes.cdll.LoadLibrary(str(_so))
            except OSError:
                pass
        break

# Clear stale compile caches
_cache_dirs_to_clear = [
    Path("/tmp/unsloth_compiled_cache"),
    Path.cwd() / "unsloth_compiled_cache",
    Path(__file__).parent.parent.parent / "unsloth_compiled_cache",
]
for _cache_dir in _cache_dirs_to_clear:
    if _cache_dir.exists():
        try:
            shutil.rmtree(_cache_dir)
        except Exception:
            pass


def patch_torch_compile_wrapper() -> None:
    """Patch torch's inductor wrapper to ignore unknown option keys.

    Without this patch, Unsloth's calls to ``torch.compile`` may pass
    configuration options that the installed torch version does not
    recognise, causing a ``RuntimeError``.
    """
    try:
        import torch
        wrapper_class = getattr(torch, "_TorchCompileInductorWrapper", None)
        if wrapper_class is None:
            return
        _original_apply_options = wrapper_class.apply_options

        def _patched_apply_options(self, options):  # type: ignore[no-untyped-def]
            if not options:
                return
            from torch._inductor import config as inductor_config

            def get_known_keys(cfg, prefix=""):  # type: ignore[no-untyped-def]
                keys: set[str] = set()
                for key in dir(cfg):
                    if key.startswith("_"):
                        continue
                    full_key = f"{prefix}{key}" if prefix else key
                    val = getattr(cfg, key, None)
                    if hasattr(val, "__dict__") and not callable(val) and not isinstance(val, (str, int, float, bool, list, dict, type(None))):
                        keys.update(get_known_keys(val, f"{full_key}."))
                    else:
                        keys.add(full_key)
                return keys

            try:
                known_keys = get_known_keys(inductor_config)
            except Exception:
                known_keys = set()
            filtered_options = {k: v for k, v in options.items() if k in known_keys}
            if filtered_options:
                try:
                    _original_apply_options(self, filtered_options)
                except RuntimeError:
                    pass

        wrapper_class.apply_options = _patched_apply_options
    except Exception:
        pass


# Apply the patch immediately on import
patch_torch_compile_wrapper()

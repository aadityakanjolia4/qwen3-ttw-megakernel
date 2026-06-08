"""Preload CUDA shared libs from pip-installed nvidia-* packages.

Invoked at Python startup via a .pth file installed by `pip install -e .`
Must run BEFORE torch is imported so the dynamic linker can find libcublas etc.
Fails silently — never crashes the interpreter.
"""
import ctypes
import os
import pathlib


def _preload() -> None:
    pkg_names = ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime", "nvidia.cufft", "nvidia.curand")
    lib_dirs: list[str] = []

    for name in pkg_names:
        try:
            mod = __import__(name, fromlist=["__file__"])
            lib_dir = pathlib.Path(mod.__file__).parent / "lib"
            if not lib_dir.is_dir():
                continue
            lib_dirs.append(str(lib_dir))
            for so in sorted(lib_dir.glob("*.so*")):
                if so.is_symlink():
                    continue
                try:
                    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        except (ImportError, Exception):
            pass

    if lib_dirs:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (":" + existing if existing else "")


_preload()

"""setup.py — exists only to hook pip install -e . and write a CUDA preload .pth file."""

import os
import site

from setuptools import setup
from setuptools.command.egg_info import egg_info

PTH_NAME = "qwen3_cuda_preload.pth"
PTH_CONTENT = "import qwen_tts._cuda_preload\n"


def _write_pth() -> None:
    try:
        candidates = list(site.getsitepackages())
    except Exception:
        candidates = []
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass

    for sp in candidates:
        pth_path = os.path.join(sp, PTH_NAME)
        try:
            os.makedirs(sp, exist_ok=True)
            with open(pth_path, "w") as f:
                f.write(PTH_CONTENT)
            print(f"[setup] CUDA preload hook installed → {pth_path}")
            return
        except (OSError, PermissionError):
            continue

    print("[setup] Warning: could not write CUDA preload .pth (check site-packages permissions)")


class EggInfoCommand(egg_info):
    """Extend egg_info so our hook runs on both old and new pip editable installs."""

    def run(self):
        super().run()
        _write_pth()


setup(cmdclass={"egg_info": EggInfoCommand})

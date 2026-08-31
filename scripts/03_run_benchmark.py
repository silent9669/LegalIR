"""Compatibility entrypoint forwarding to 05_run_oof.py."""

import importlib.util
from pathlib import Path
import sys

_script = Path(__file__).resolve().parent / "05_run_oof.py"
_spec = importlib.util.spec_from_file_location("run_oof_05", _script)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_evaluation = _mod.run_oof_validation
main = _mod.main

if __name__ == "__main__":
    main()

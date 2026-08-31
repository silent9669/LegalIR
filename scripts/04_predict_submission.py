"""Compatibility entrypoint forwarding to 07_predict_submission.py."""

import importlib.util
from pathlib import Path
import sys

_script = Path(__file__).resolve().parent / "07_predict_submission.py"
_spec = importlib.util.spec_from_file_location("predict_submission_07", _script)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

generate_submission = _mod.generate_submission
main = _mod.main

if __name__ == "__main__":
    main()

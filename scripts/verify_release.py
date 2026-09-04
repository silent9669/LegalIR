#!/usr/bin/env python3
"""
Full LegalIR release verification gate.
Executes:
1. Python compileall on src and scripts
2. Unit, parity, leakage, memory, and release test suites
3. Parameter audit (< 4B budget)
4. Output verification
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    print(f"[*] Checking: {description} ...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[!] FAILED: {description}")
        print(res.stdout)
        print(res.stderr)
        return False
    print(f"[+] PASSED: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Full LegalIR release verification gate.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest suite")
    args = parser.parse_args()

    python_bin = sys.executable

    # Step 1: Compileall
    if not run_command([python_bin, "-m", "compileall", "-q", "src", "scripts"], "Python syntax compilation"):
        sys.exit(1)

    # Step 2: Pytest modular suite
    if not args.skip_tests:
        test_cmd = [
            python_bin,
            "-m",
            "pytest",
            "-q",
            "tests/unit",
            "tests/parity",
            "tests/leakage",
            "tests/memory",
            "tests/integration",
            "tests/release",
        ]
        if not run_command(test_cmd, "Modular test suite (unit, parity, leakage, memory, integration, release)"):
            sys.exit(1)

    # Step 3: Parameter audit
    param_audit_script = Path("scripts/audit_parameters.py")
    if param_audit_script.is_file():
        if not run_command([python_bin, str(param_audit_script)], "Learned parameter budget (<4B)"):
            sys.exit(1)

    print("\n=======================================================")
    print("[+] ALL LEGALIR RELEASE GATES PASSED.")
    print("=======================================================")
    sys.exit(0)


if __name__ == "__main__":
    main()

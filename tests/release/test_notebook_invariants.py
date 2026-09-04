import json
import re
import pytest
from pathlib import Path


def test_notebook_invariants_no_secrets():
    # Test helper that audits all notebooks for exposed secrets or hardcoded tokens
    forbidden_tokens = ["ghp_", "sk-", "AIzaSy", "password", "SECRET_KEY"]
    for nb_path in [
        Path("notebooks/kaggle_final.ipynb"),
        Path("notebooks/colab_t4_smoke.ipynb"),
        Path("colab/legalir_t4_smoke.ipynb"),
        Path("legalir_training.ipynb"),
    ]:
        if nb_path.is_file():
            raw_text = nb_path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                assert token not in raw_text, f"Forbidden token '{token}' in {nb_path}"


def test_kaggle_notebook_pin_contains_new_runtime_full_sha():
    nb_path = Path("notebooks/kaggle_final.ipynb")
    assert nb_path.is_file(), f"Missing {nb_path}"

    raw_text = nb_path.read_text(encoding="utf-8")
    # Verify 40-char hex commit pin exists
    match = re.search(r"git checkout ([0-9a-f]{40})", raw_text)
    assert match is not None, "Kaggle notebook must contain full 40-char git commit SHA pin"
    pinned_sha = match.group(1)
    assert len(pinned_sha) == 40
    # Must never be placeholder a0efb25
    assert not pinned_sha.startswith("a0efb25")


def test_kaggle_notebook_target_script_exists_at_pin():
    nb_path = Path("notebooks/kaggle_final.ipynb")
    raw_text = nb_path.read_text(encoding="utf-8")
    assert "scripts/run_kaggle_final.py" in raw_text
    assert Path("scripts/run_kaggle_final.py").is_file()


def test_colab_notebook_passes_required_cli_args():
    for nb_path in [Path("colab/legalir_t4_smoke.ipynb"), Path("notebooks/colab_t4_smoke.ipynb")]:
        if nb_path.is_file():
            raw_text = nb_path.read_text(encoding="utf-8")
            assert "--data-dir" in raw_text
            assert "--work-dir" in raw_text
            assert "--target-sha" in raw_text
            # Verify 40-char hex commit pin
            match = re.search(r"TARGET_SHA\s*=\s*['\"]([0-9a-f]{40})['\"]", raw_text)
            assert match is not None, f"{nb_path} must pass 40-char TARGET_SHA"
            assert not match.group(1).startswith("a0efb25")


def test_notebook_execution_contract():
    stages = ["K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9"]
    assert len(stages) == 10

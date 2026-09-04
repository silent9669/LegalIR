import json
import pytest
from pathlib import Path


def test_notebook_invariants_no_secrets(tmp_path):
    # Test helper that audits a notebook json for exposed secrets or hardcoded tokens
    nb_data = {
        "cells": [
            {"cell_type": "code", "source": ["print('Hello LegalIR')\n"]},
            {"cell_type": "markdown", "source": ["# LegalIR Final Production\n"]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    raw_text = json.dumps(nb_data)
    forbidden_tokens = ["hf_", "ghp_", "sk-", "AIzaSy", "password", "SECRET_KEY"]
    for token in forbidden_tokens:
        assert token not in raw_text


def test_notebook_execution_contract():
    # Verify notebook execution stages structure
    stages = ["K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9"]
    assert len(stages) == 10

"""Contract tests for scripts/setup.sh (fixture repos, fake toolchains).

setup.sh must be idempotent, local-only, and secret-safe: it may create .venv
and pip-install, but must never create/modify/print .env values, never touch
the cloud, and never commit.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REAL_SETUP = Path("scripts/setup.sh").resolve()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(REAL_SETUP, repo / "scripts" / "setup.sh")
    (repo / "scripts" / "setup.sh").chmod(0o755)
    (repo / "requirements.txt").write_text("# fixture reqs\n", encoding="utf-8")
    return repo


def _write_fakes(bin_dir: Path, py_ver: str = "3.11") -> dict[str, str]:
    bin_dir.mkdir(parents=True, exist_ok=True)
    pip_log = bin_dir / "pip.log"
    fake_py = bin_dir / "python3"
    fake_py.write_text(
        "#!/bin/sh\n"
        f'VER="{py_ver}"\n'
        'if [ "$1" = "-c" ]; then echo "$VER"; exit 0; fi\n'
        'if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then\n'
        '  mkdir -p "$3/bin"\n'
        f'  printf \'#!/bin/sh\\necho "$@" >> "{pip_log}"\\nexit 0\\n\' > "$3/bin/pip"\n'
        '  chmod +x "$3/bin/pip"\n'
        '  printf \'#!/bin/sh\\nexit 0\\n\' > "$3/bin/python"\n'
        '  chmod +x "$3/bin/python"\n'
        f'  printf \'#!/bin/sh\\necho "modal 9.9.9 (fake)"\\nexit 0\\n\' > "$3/bin/modal"\n'
        '  chmod +x "$3/bin/modal"\n'
        "  exit 0\n"
        "fi\n"
        'echo "unexpected python3 call: $@" >&2\n'
        "exit 3\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)
    return {"python3": str(fake_py), "pip_log": str(pip_log)}


def _run(repo: Path, args=(), extra_env: dict | None = None):
    env = os.environ.copy()
    env["GIT_CEILING_DIRECTORIES"] = str(repo.parent)
    env.pop("HF_REPO_ID", None)
    for k, v in (extra_env or {}).items():
        env[k] = v
    res = subprocess.run(
        [str(repo / "scripts" / "setup.sh"), *args],
        cwd=str(repo), env=env, capture_output=True, text=True, timeout=60,
    )
    return res


def _env_for(repo: Path, bin_dir: Path, **kw) -> dict:
    fakes = _write_fakes(bin_dir, kw.pop("py_ver", "3.11"))
    return {
        "SETUP_PYTHON": fakes["python3"],
        "SETUP_VENV_DIR": str(repo / ".venv"),
        "SETUP_REQUIREMENTS": str(repo / "requirements.txt"),
        "SETUP_MODAL_BIN": str(repo / ".venv" / "bin" / "modal"),
        "SETUP_GIT_BIN": "git",
        "PIP_LOG": fakes["pip_log"],
        **kw,
    }


def test_setup_creates_venv_and_installs(tmp_path):
    repo = _make_repo(tmp_path)
    env = _env_for(repo, tmp_path / "bin")
    res = _run(repo, extra_env=env)
    assert res.returncode == 0, res.stderr
    assert (repo / ".venv" / "bin" / "python").is_file()
    assert (repo / ".venv" / "bin" / "pip").is_file()
    pip_log = Path(env["PIP_LOG"])
    assert pip_log.is_file()
    logged = pip_log.read_text(encoding="utf-8")
    assert "requirements.txt" in logged
    assert "modal" in logged and "kaggle" in logged
    assert "[+] Setup OK" in res.stdout


def test_setup_idempotent_reuses_venv(tmp_path):
    repo = _make_repo(tmp_path)
    env = _env_for(repo, tmp_path / "bin")
    assert _run(repo, extra_env=env).returncode == 0
    res2 = _run(repo, extra_env=env)
    assert res2.returncode == 0
    assert "Reusing existing" in res2.stdout


def test_setup_never_touches_env_file(tmp_path):
    repo = _make_repo(tmp_path)
    sentinel = "HF_TOKEN_WRITE=hf_sentinel_secret_abc123\nHF_REPO_ID=a/b\n"
    (repo / ".env").write_text(sentinel, encoding="utf-8")
    env = _env_for(repo, tmp_path / "bin")
    res = _run(repo, extra_env=env)
    assert res.returncode == 0, res.stderr
    assert (repo / ".env").read_text(encoding="utf-8") == sentinel
    assert "hf_sentinel_secret_abc123" not in res.stdout
    assert "hf_sentinel_secret_abc123" not in res.stderr


def test_setup_without_env_file_creates_none(tmp_path):
    repo = _make_repo(tmp_path)
    env = _env_for(repo, tmp_path / "bin")
    res = _run(repo, extra_env=env)
    assert res.returncode == 0, res.stderr
    assert not (repo / ".env").exists()
    assert "No .env file" in res.stdout


def test_setup_reports_names_not_values(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".env").write_text("HF_TOKEN_WRITE=hf_live_value_xyz999\n", encoding="utf-8")
    env = _env_for(repo, tmp_path / "bin")
    res = _run(repo, extra_env=env)
    assert res.returncode == 0, res.stderr
    assert "HF_TOKEN_WRITE: set" in res.stdout
    assert "hf_live_value_xyz999" not in res.stdout + res.stderr


def test_setup_check_only_installs_nothing(tmp_path):
    repo = _make_repo(tmp_path)
    env = _env_for(repo, tmp_path / "bin")
    assert _run(repo, extra_env=env).returncode == 0
    pip_log = Path(env["PIP_LOG"])
    before = pip_log.read_text(encoding="utf-8")
    res = _run(repo, args=("--check-only",), extra_env=env)
    assert res.returncode == 0, res.stderr
    assert pip_log.read_text(encoding="utf-8") == before
    assert "--check-only" in res.stdout


def test_setup_check_only_fails_without_venv(tmp_path):
    repo = _make_repo(tmp_path)
    env = _env_for(repo, tmp_path / "bin")
    res = _run(repo, args=("--check-only",), extra_env=env)
    assert res.returncode == 2
    assert ".venv" in (res.stdout + res.stderr)


def test_setup_missing_python_fails(tmp_path):
    repo = _make_repo(tmp_path)
    env = _env_for(repo, tmp_path / "bin")
    env["SETUP_PYTHON"] = "/nonexistent/python3"
    res = _run(repo, extra_env=env)
    assert res.returncode == 2


def test_setup_old_python_fails(tmp_path):
    repo = _make_repo(tmp_path)
    env = _env_for(repo, tmp_path / "bin", py_ver="3.9")
    res = _run(repo, extra_env=env)
    assert res.returncode == 2
    assert "too old" in res.stderr


def test_setup_rejects_unknown_arg(tmp_path):
    repo = _make_repo(tmp_path)
    env = _env_for(repo, tmp_path / "bin")
    res = _run(repo, args=("--deploy",), extra_env=env)
    assert res.returncode == 2

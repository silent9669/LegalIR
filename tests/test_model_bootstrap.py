from pathlib import Path
from src.models.bootstrap import required_model_files
from src.models.device import resolve_device


def test_required_files_exclude_onnx_and_openvino():
    files = required_model_files("BAAI/bge-m3")
    assert "pytorch_model.bin" in files or "*.safetensors" in files
    assert not any(name.startswith("onnx/") for name in files)
    assert not any(name.startswith("openvino/") for name in files)


def test_explicit_cpu_is_preserved():
    assert resolve_device("cpu") == "cpu"


def test_auto_device_resolves():
    dev = resolve_device("auto")
    assert dev in {"mps", "cuda", "cpu"}

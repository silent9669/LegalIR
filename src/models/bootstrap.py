from pathlib import Path
from typing import Any
import argparse
import json
from huggingface_hub import snapshot_download
from src.core.config import load_pipeline_config
from src.core.paths import ProjectPaths

MODEL_REGISTRY = {
    "BAAI/bge-m3": {
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "allow_patterns": ["*.json", "*.txt", "pytorch_model.bin", "*.safetensors", "sentencepiece.bpe.model", "tokenizer*"],
        "ignore_patterns": ["onnx/*", "openvino/*", "*.msgpack", "*.h5", "coreml/*"],
    },
    "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2": {
        "revision": "99a2963b2f51fa7a570a3e7f550d7993b9de90a8",
        "allow_patterns": ["*.json", "*.txt", "pytorch_model.bin", "*.safetensors", "sentencepiece.bpe.model", "bpe.codes", "tokenizer*"],
        "ignore_patterns": ["onnx/*", "openvino/*", "*.msgpack", "*.h5", "coreml/*"],
    },
    "BAAI/bge-reranker-v2-m3": {
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "allow_patterns": ["*.json", "*.txt", "pytorch_model.bin", "*.safetensors", "sentencepiece.bpe.model", "tokenizer*"],
        "ignore_patterns": ["onnx/*", "openvino/*", "*.msgpack", "*.h5", "coreml/*"],
    },
}


def required_model_files(model_name: str) -> list[str]:
    reg = MODEL_REGISTRY.get(model_name, {})
    return reg.get("allow_patterns", ["*.json", "*.txt", "*.bin", "*.safetensors"])


def download_models(
    config: dict[str, Any] | None = None,
    model_root: str | Path = "artifacts/local/models/huggingface",
    local_files_only: bool = False,
) -> dict[str, Path]:
    model_root = Path(model_root)
    model_root.mkdir(parents=True, exist_ok=True)

    downloaded_paths = {}

    for model_id, meta in MODEL_REGISTRY.items():
        print(f"\nDownloading/verifying pinned model {model_id} (revision {meta['revision'][:8]})...")
        path = snapshot_download(
            repo_id=model_id,
            revision=meta["revision"],
            cache_dir=str(model_root),
            allow_patterns=meta["allow_patterns"],
            ignore_patterns=meta["ignore_patterns"],
            local_files_only=local_files_only,
        )
        downloaded_paths[model_id] = Path(path)
        print(f"Model {model_id} ready at {path}")

    manifest_path = model_root / "manifest.json"
    manifest_data = {
        model_id: {
            "path": str(p),
            "revision": MODEL_REGISTRY[model_id]["revision"],
        }
        for model_id, p in downloaded_paths.items()
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    return downloaded_paths


def main():
    parser = argparse.ArgumentParser(description="LegalIR Pinned Local Model Bootstrap CLI")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--local-only", action="store_true", help="Only use local cached files")
    args = parser.parse_args()

    paths = ProjectPaths.from_repo()
    cfg = {}
    if Path(args.config).exists():
        cfg = load_pipeline_config(Path(args.config))

    model_root = paths.local_models / "huggingface"
    download_models(config=cfg, model_root=model_root, local_files_only=args.local_only)


if __name__ == "__main__":
    main()

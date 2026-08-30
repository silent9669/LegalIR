import torch
from transformers import AutoModel, AutoModelForSequenceClassification

def audit_parameters():
    print("=" * 60)
    print("UIT-DSC 2026 Task 1: System Learned Parameter Audit")
    print("=" * 60)

    # 1. DEk21 Dense Retriever
    print("\nLoading DEk21 v2...")
    dek21 = AutoModel.from_pretrained("CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2")
    dek21_params = sum(p.numel() for p in dek21.parameters())
    print(f"DEk21 v2 Parameters: {dek21_params:,} ({dek21_params / 1e9:.4f}B)")

    # 2. BGE Reranker v2 M3
    print("\nLoading BGE Reranker v2 M3...")
    bge_reranker = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
    reranker_params = sum(p.numel() for p in bge_reranker.parameters())
    print(f"BGE Reranker v2 M3 Parameters: {reranker_params:,} ({reranker_params / 1e9:.4f}B)")

    total_params = dek21_params + reranker_params
    print("\n" + "-" * 60)
    print(f"TOTAL SYSTEM PARAMETERS: {total_params:,} ({total_params / 1e9:.4f}B)")
    print(f"COMPETITION LIMIT:      4,000,000,000 (4.0000B)")

    if total_params < 4_000_000_000:
        print(f"COMPLIANCE VERDICT:      PASS ({total_params / 1e9:.4f}B < 4.0B)")
    else:
        print("COMPLIANCE VERDICT:      FAIL (Exceeds 4.0B limit)")
    print("=" * 60)

if __name__ == "__main__":
    audit_parameters()

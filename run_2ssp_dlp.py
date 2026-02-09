"""
2SSP + DLP 결합 프루닝 실행 스크립트
- 채널 선택: 2SSP L2 norm (hidden state)
- 레이어별 sparsity: DLP 비율 조정
"""

import argparse
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# 프로젝트 루트
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

def get_calibration_data(tokenizer, nsamples=32, seqlen=2048, seed=0):
    """2SSP 스타일 calibration 데이터 생성 (C4 또는 wikitext)"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    try:
        traindata = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    except Exception:
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    random = __import__("random")
    random.seed(seed)
    calibration = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            text = traindata[i]["text"] if "text" in traindata[i] else traindata[i]["sentence"]
            enc = tokenizer(text, return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        calibration.append(enc.input_ids[:, i : i + seqlen])

    return calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="모델 경로 (예: Qwen/Qwen3-8B, meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--pruning_rate", type=float, default=0.5,
                        help="전체 목표 sparsity (0.5 = 50% 파라미터 제거)")
    parser.add_argument("--alpha", type=float, default=1.5,
                        help="Attention vs MLP 비율 조정 (2SSP, 기본 1.5)")
    parser.add_argument("--alpha_dlp", type=float, default=0.15,
                        help="레이어별 sparsity 스케일링 (DLP)")
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--prune_attention", type=int, default=None,
                        help="Attention 제거 개수 (None=alpha로 자동 계산, 0=MLP만)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading model...")
    _model_lower = args.model.lower()
    dtype = torch.bfloat16 if any(x in _model_lower for x in ("llama", "qwen", "mistral")) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        use_cache=False,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    model.eval()

    print("Preparing calibration data...")
    calibration_dataset = get_calibration_data(
        tokenizer, nsamples=args.nsamples, seqlen=2048, seed=args.seed
    )

    print("Running 2SSP+DLP pruning...")
    from prune_2ssp_dlp import prune_mlp_2ssp_dlp
    model = prune_mlp_2ssp_dlp(
        model,
        calibration_dataset,
        pruning_rate=args.pruning_rate,
        alpha=args.alpha,
        alpha_dlp=args.alpha_dlp,
        num_attn_submodules_to_prune=args.prune_attention,
    )

    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"Model saved to {args.save_model}")

    print("Done.")


if __name__ == "__main__":
    main()

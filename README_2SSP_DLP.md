# 2SSP + DLP 결합 프루닝

## 개요

- **2SSP**: 어떤 채널을 자를지 결정 (MLP hidden state L2 norm 기반)
- **DLP**: 레이어별로 다른 sparsity 비율 적용 (layer-wise ratio adjustment)
- **결합**: 전체 목표 sparsity에서 Attention vs MLP 비율 조정(alpha) + DLP 레이어별 분배

## 알고리즘

1. **전체 압축률 통합** (`pruning_rate` = 전체 목표 sparsity)
   - `alpha`로 Attention vs MLP 비율 조정 (2SSP Equation 5)
   - `num_attn = round(num_blocks * pow(pruning_rate, (mlp/attn)/alpha))`
   - 나머지는 MLP에서 제거

2. **채널 선택 (2SSP 기준)**
   - 각 레이어 MLP hidden state L2 norm
   - norm 낮은 채널 = 제거, norm 높은 채널 = 보존

3. **레이어별 분배 (DLP + alpha_dlp)**
   - 덜 중요한 레이어 → 더 많이 prune
   - `get_dlp_ratios_2ssp()`: L2 norm 기반 레이어 점수 + DLP scaling

## 파라미터

| 파라미터 | 역할 | 기본값 |
|----------|------|--------|
| `pruning_rate` | 전체 목표 sparsity | 0.5 |
| `alpha` | **Attention vs MLP 비율** (2SSP) | 1.5 |
| `alpha_dlp` | 레이어별 sparsity 스케일링 (DLP) | 0.15 |

- `alpha` ↑ → Attention 더 많이 제거
- `alpha` ↓ → MLP에 더 분배

## 사용법

```bash
# 기본 (alpha로 Attention/MLP 비율 자동 계산)
python run_2ssp_dlp.py \
  --model meta-llama/Llama-2-7b-hf \
  --pruning_rate 0.5 \
  --alpha 1.5 \
  --save_model pruned/llama2-7b-2ssp-dlp

# MLP만 (Attention 제거 없음)
python run_2ssp_dlp.py --pruning_rate 0.5 --prune_attention 0

# Attention 개수 수동 지정
python run_2ssp_dlp.py --pruning_rate 0.5 --prune_attention 2
```

## 코드 구조

- `prune_2ssp_dlp.py`: 핵심 결합 로직
  - `get_dlp_ratios_2ssp()`: 2SSP L2 norm 기반 레이어별 비율
  - `prune_mlp_2ssp_dlp()`: 2SSP+DLP 결합 MLP 채널 프루닝
- `run_2ssp_dlp.py`: 실행 스크립트
- `_2ssp_src/`: 2SSP 유틸 내장 (서브모듈 없이 단일 repo)

## 2SSP vs DLP vs 결합

| 구분 | 2SSP | DLP | 2SSP+DLP |
|------|------|-----|----------|
| 채널 선택 | L2 norm (hidden) | Wanda (|W|·√act) | L2 norm |
| 레이어별 sparsity | 동일 (uniform) | 상이 (layer-wise) | 상이 |
| 적용 대상 | MLP 채널 + Attention | 가중치 (unstructured) | MLP 채널 |

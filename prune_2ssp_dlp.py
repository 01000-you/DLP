"""
2SSP + DLP 결합 프루닝
- 채널 선택: 2SSP의 L2 norm (hidden state 기반) - 어떤 채널을 자를지
- 레이어별 sparsity: DLP의 layer-wise 비율 조정 - 각 레이어마다 다른 sparsity
"""

import logging
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

# 2SSP 유틸 (내장)

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from _2ssp_src.utilities import get_mlp_hidden_state, prune_mlp
from _2ssp_src.block_metrics import block_influence


def get_mlp_hidden_state_batch(model, calibration_samples, device):
    """
    2SSP 스타일: 여러 calibration 샘플에 대해 MLP hidden state 수집
    calibration_samples: list of tensors (1, seq_len) each
    Returns: dict layer_idx -> list of tensors (seq_len, intermediate_size)
    """
    hidden_states_list = {}  # layer_idx -> list of tensors
    
    for sample in calibration_samples:
        if model.config.model_type in ("llama", "mistral", "phi3", "qwen2", "qwen3"):
            for i, layer in enumerate(model.model.layers):
                layer.mlp.down_proj.original_index = i
        else:
            for i, layer in enumerate(model.model.layers):
                layer.mlp.fc2.original_index = i

        hidden_states = {}

        def hook(module, input, output):
            hidden_states[module.original_index] = input[0][0].to("cpu")

        hooks = []
        for layer in model.model.layers:
            if model.config.model_type in ("llama", "mistral", "phi3", "qwen2", "qwen3"):
                last_linear = layer.mlp.down_proj
            else:
                last_linear = layer.mlp.fc2
            hooks.append(last_linear.register_forward_hook(
                lambda m, inp, out: hook(m, inp, out)
            ))

        input_ids = sample.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            _ = model(input_ids)

        for h in hooks:
            h.remove()

        for li, ten in hidden_states.items():
            if li not in hidden_states_list:
                hidden_states_list[li] = []
            hidden_states_list[li].append(ten)

    return hidden_states_list


def get_dlp_ratios_2ssp(model, calibration_dataset, sparsity_ratio, alpha_dlp=0.15):
    """
    DLP 스타일 레이어별 비율 조정 + 2SSP L2 norm 기반 레이어 중요도
    - 각 레이어의 MLP hidden state L2 norm 평균으로 레이어 중요도 계산
    - 덜 중요한 레이어는 더 많이 prune (높은 sparsity)
    alpha_dlp: 레이어별 sparsity 스케일링 (DLP)
    Returns: list of imp_ratio (보존 비율) per layer, 길이 = num_layers
    """
    num_blocks = len(model.model.layers)
    mlp_hidden_size = model.config.intermediate_size

    # 2SSP 방식: 각 레이어별 채널 L2 norm 평균
    average_norms = [torch.zeros(mlp_hidden_size) for _ in range(num_blocks)]

    for sample in tqdm(calibration_dataset, desc="Computing layer scores"):
        hidden_states_ci = get_mlp_hidden_state(model, sample)
        for li in range(num_blocks):
            if isinstance(hidden_states_ci, dict):
                hs = hidden_states_ci[li]
            else:
                hs = hidden_states_ci[li]
            norm_ci_li = hs.norm(dim=0, p=2)
            average_norms[li] = average_norms[li].to(norm_ci_li.device) + norm_ci_li

    for li in range(num_blocks):
        average_norms[li] /= len(calibration_dataset)

    # 레이어별 중요도 = 평균 L2 norm (높을수록 중요한 레이어)
    layer_score = [torch.mean(avg_norm).item() for avg_norm in average_norms]

    # DLP 방식: ratio_conn = 1 - layer/total (덜 중요 = 높은 ratio = 더 많이 prune)
    total_conn = sum(layer_score)
    if total_conn < 1e-8:
        total_conn = 1.0  # Avoid division by zero
    ratio_conn = [1 - ls / total_conn for ls in layer_score]

    imp_ratios = torch.tensor(ratio_conn)
    min_ratio = torch.min(imp_ratios)
    max_ratio = torch.max(imp_ratios)
    if max_ratio - min_ratio < 1e-8:
        scaled_ratios = torch.ones_like(imp_ratios) * (1 - sparsity_ratio)
    else:
        scaled_ratios = (imp_ratios - min_ratio) * (1 / (max_ratio - min_ratio) * alpha_dlp * 2)
    all_layer_ratio = (scaled_ratios - torch.mean(scaled_ratios) + (1 - sparsity_ratio)).tolist()
    
    # Check for NaN values and replace with defaults
    all_layer_ratio = [float(r) if not (isinstance(r, float) and (r != r)) else (1 - sparsity_ratio) for r in all_layer_ratio]

    log.info(f"    layer_score 범위: {min(layer_score):.4f} ~ {max(layer_score):.4f}, imp_ratio 범위: {min(all_layer_ratio):.4f} ~ {max(all_layer_ratio):.4f}")

    return all_layer_ratio, average_norms


@torch.no_grad()
def _get_cfsp_layer_importances(model, calibration_dataset, device, metrics="angular"):
    """
    캘리브레이션 샘플별로 전체 forward 하며 각 레이어의 (입력, 출력)을 수집하고,
    block_influence 합으로 레이어별 중요도 리스트 반환.
    """
    layers = model.model.layers
    num_blocks = len(layers)
    layer_importances = [0.0] * num_blocks

    for sample in tqdm(calibration_dataset, desc="CFSP layer importance"):
        layer_io = {}  # layer_idx -> (inp, out)

        def make_hook(idx):
            def hook(module, input, output):
                inp = input[0].detach()
                out = output[0] if isinstance(output, tuple) else output
                out = out.detach()
                layer_io[idx] = (inp, out)
            return hook

        hooks = []
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

        ids = sample.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        _ = model(ids)

        for h in hooks:
            h.remove()

        for i in range(num_blocks):
            inp, out = layer_io[i]
            layer_importances[i] += block_influence(inp, out, metrics=metrics).sum().cpu().item()

    return layer_importances


def get_cfsp_ratios_2ssp(
    model,
    calibration_dataset,
    sparsity_ratio,
    metrics="angular",
    sigmoid_a=1.0,
    device=None,
):
    """
    CFSP 스타일: 레이어 입력-출력 거리(민감도)로 보존 비율 계산 + 2SSP L2 norm.
    - layer_importances: block_influence 합 (클수록 중요한 레이어)
    - mid 기준 정규화 후 sigmoid, every_pruning_ratios = x/avg * (1 - sparsity_ratio)
    - 2SSP 채널 선택용 average_norms는 DLP와 동일(MLP hidden L2 norm).
    Returns: (imp_ratios, average_norms) — get_dlp_ratios_2ssp와 동일 형식.
    """
    if device is None:
        device = next(model.parameters()).device
    num_blocks = len(model.model.layers)
    mlp_hidden_size = model.config.intermediate_size

    # 레이어별 CFSP 중요도 (입력-출력 거리)
    layer_importances = _get_cfsp_layer_importances(
        model, calibration_dataset, device, metrics=metrics
    )

    # 2SSP용 레이어별 채널 L2 norm (DLP와 동일)
    average_norms = [torch.zeros(mlp_hidden_size) for _ in range(num_blocks)]
    for sample in tqdm(calibration_dataset, desc="CFSP L2 norms"):
        hidden_states_ci = get_mlp_hidden_state(model, sample)
        for li in range(num_blocks):
            hs = hidden_states_ci[li]
            norm_li = hs.norm(dim=0, p=2)
            average_norms[li] = average_norms[li].to(norm_li.device) + norm_li
    for li in range(num_blocks):
        average_norms[li] /= len(calibration_dataset)

    # CFSP: mid 기준 정규화 후 sigmoid
    mid = sum(layer_importances) / len(layer_importances)
    normalized = [(x - mid) / 1e4 for x in layer_importances]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x * sigmoid_a))

    layer_weights = [sigmoid(x) for x in normalized]
    avg = sum(layer_weights) / len(layer_weights)
    max_score = max(layer_weights)

    # max/avg * (1 - pruning_ratio) >= 1 이면 scale_factor로 조정
    if avg > 1e-8 and max_score / avg * (1 - sparsity_ratio) >= 1:
        scale_factor = (avg * (1 / (1 - sparsity_ratio) - 1)) / (max_score - avg + 1e-8) / 1.05
        scale_factor = max(0.0, min(1.0, scale_factor))
        for i in range(len(layer_weights)):
            if layer_weights[i] > avg:
                layer_weights[i] = avg + (layer_weights[i] - avg) * scale_factor
            else:
                layer_weights[i] = avg - (avg - layer_weights[i]) * scale_factor
        avg = sum(layer_weights) / len(layer_weights)

    imp_ratios = [x / avg * (1 - sparsity_ratio) for x in layer_weights]
    imp_ratios = [
        float(r) if not (isinstance(r, float) and (r != r)) else (1 - sparsity_ratio)
        for r in imp_ratios
    ]
    log.info(
        f"    CFSP layer_importances 범위: {min(layer_importances):.4f} ~ {max(layer_importances):.4f}, "
        f"imp_ratio 범위: {min(imp_ratios):.4f} ~ {max(imp_ratios):.4f}"
    )
    return imp_ratios, average_norms


def prune_mlp_2ssp_dlp(
    model,
    calibration_dataset,
    pruning_rate,
    alpha=1.5,
    alpha_dlp=0.15,
    num_attn_submodules_to_prune=None,
    layer_sparsity_method="dlp",
    cfsp_metrics="angular",
    cfsp_sigmoid_a=1.0,
):
    """
    2SSP + DLP/CFSP 통합: 전체 목표 sparsity에서 Attention vs MLP 비율 조정 후 MLP 레이어별 프루닝

    - pruning_rate: 전체 목표 sparsity (0.5 = 50% 파라미터 제거)
    - alpha: Attention vs MLP 비율 조정 (2SSP Equation 5, default 1.5)
    - alpha_dlp: 레이어별 sparsity 스케일링 (DLP; layer_sparsity_method="dlp"일 때만 사용)
    - num_attn_submodules_to_prune: None이면 alpha로 계산
    - layer_sparsity_method: "dlp" | "cfsp" — 레이어별 보존 비율 결정 방식
    - cfsp_metrics: CFSP 시 block_influence 메트릭 ("angular" | "cosine" | "mse" | "mae")
    - cfsp_sigmoid_a: CFSP sigmoid 스케일 인자
    """
    layers = model.model.layers
    num_blocks = len(layers)
    mlp_hidden_size = model.config.intermediate_size

    log.info("  [Stage 0] 모델 구조 분석")
    original_total_params = sum(p.numel() for p in model.parameters())
    main_model_total_params = sum(p.numel() for p in model.model.layers.parameters())
    attn_total_params = sum(p.numel() for p in model.model.layers[0].self_attn.parameters())
    mlp_total_params = sum(p.numel() for p in model.model.layers[0].mlp.parameters())
    log.info(f"    layers: {num_blocks}, hidden: {model.config.hidden_size}, intermediate: {mlp_hidden_size}")
    log.info(f"    params - 전체(임베딩 포함): {original_total_params/1e6:.2f}M, 레이어만: {main_model_total_params/1e6:.2f}M, attn/layer: {attn_total_params/1e6:.2f}M, mlp/layer: {mlp_total_params/1e6:.2f}M")

    # 1. Attention vs MLP 비율 (alpha로 조정)
    log.info("  [Stage 1] Attention vs MLP 비율 계산 (alpha)")
    if num_attn_submodules_to_prune is None:
        num_attn_submodules_to_prune = round(
            num_blocks * pow(pruning_rate, (mlp_total_params / attn_total_params) / alpha)
        )
    num_attn_submodules_to_prune = max(0, min(num_attn_submodules_to_prune, num_blocks))
    log.info(f"    num_attn_to_prune: {num_attn_submodules_to_prune} (alpha={alpha})")

    # 2. 전체 목표에 맞춰 MLP에서 제거할 파라미터 수
    parameters_pruned_for_attention = num_attn_submodules_to_prune * attn_total_params
    target_parameters_to_prune = int(round(pruning_rate * main_model_total_params))
    total_mlp_params_to_prune = target_parameters_to_prune - parameters_pruned_for_attention
    total_mlp_params_to_prune = max(0, total_mlp_params_to_prune)

    params_per_channel = 3 * model.config.hidden_size
    total_channels_to_prune = int(total_mlp_params_to_prune / params_per_channel)
    total_channels_to_prune = min(total_channels_to_prune, num_blocks * mlp_hidden_size - num_blocks)
    if alpha_dlp == 0:
        total_channels_to_prune = (total_channels_to_prune // num_blocks) * num_blocks

    log.info(f"    target_prune: {target_parameters_to_prune/1e6:.2f}M, attn_prune: {parameters_pruned_for_attention/1e6:.2f}M, mlp_prune: {total_mlp_params_to_prune/1e6:.2f}M")
    log.info(f"    total_channels_to_prune: {total_channels_to_prune}")

    # 3. 레이어별 비율 + 2SSP L2 norm 계산 (DLP 또는 CFSP)
    sparsity_ratio = total_channels_to_prune / (num_blocks * mlp_hidden_size)
    if layer_sparsity_method == "cfsp":
        log.info("  [Stage 2] CFSP 레이어별 비율 + 2SSP L2 norm 계산")
        imp_ratios, average_norms = get_cfsp_ratios_2ssp(
            model, calibration_dataset, sparsity_ratio,
            metrics=cfsp_metrics, sigmoid_a=cfsp_sigmoid_a,
        )
    else:
        log.info("  [Stage 2] DLP 레이어별 비율 + 2SSP L2 norm 계산")
        imp_ratios, average_norms = get_dlp_ratios_2ssp(
            model, calibration_dataset, sparsity_ratio, alpha_dlp
        )

    # 4. DLP 비율로 레이어별 채널 제거량 분배 (덜 중요한 레이어 = 더 많이 prune)
    log.info("  [Stage 3] 레이어별 채널 제거량 분배 (DLP 비율)")
    prune_weights = [1 - r for r in imp_ratios]
    # Check for NaN in prune_weights
    prune_weights = [w if not (isinstance(w, float) and (w != w)) else 1.0 for w in prune_weights]
    total_weight = sum(prune_weights)
    if total_weight < 1e-8 or total_weight != total_weight:  # Check for small or NaN
        prune_weights = [1.0 / num_blocks] * num_blocks
        total_weight = 1.0
    num_prune_per_layer = [
        int(round(total_channels_to_prune * w / total_weight))
        for w in prune_weights
    ]
    # 정수 반올림 보정 (round-robin으로 분배해 alpha_dlp=0일 때도 레이어 균등 유지)
    diff = total_channels_to_prune - sum(num_prune_per_layer)
    if diff > 0:
        for i in range(diff):
            idx = i % num_blocks
            if num_prune_per_layer[idx] < mlp_hidden_size - 1:
                num_prune_per_layer[idx] += 1
            else:
                for j in range(num_blocks):
                    k = (i + j) % num_blocks
                    if num_prune_per_layer[k] < mlp_hidden_size - 1:
                        num_prune_per_layer[k] += 1
                        break
    elif diff < 0:
        for i in range(-diff):
            idx = i % num_blocks
            if num_prune_per_layer[idx] > 0:
                num_prune_per_layer[idx] -= 1
            else:
                for j in range(num_blocks):
                    k = (i + j) % num_blocks
                    if num_prune_per_layer[k] > 0:
                        num_prune_per_layer[k] -= 1
                        break

    # 5. 각 레이어별 보존할 채널 수
    num_prune_per_layer = [min(n, mlp_hidden_size - 1) for n in num_prune_per_layer]
    num_preserve_per_layer = [
        max(1, mlp_hidden_size - n)
        for n in num_prune_per_layer
    ]

    log.info(f"    레이어별 preserve: min={min(num_preserve_per_layer)}, max={max(num_preserve_per_layer)}, mean={np.mean(num_preserve_per_layer):.0f}")
    for i in range(min(5, num_blocks)):
        log.info(f"      layer {i}: preserve={num_preserve_per_layer[i]}, prune={num_prune_per_layer[i]}")
    if num_blocks > 5:
        log.info(f"      ... (총 {num_blocks} layers)")

    # 6. 2SSP 기준으로 채널 선택 (L2 norm 상위 채널 보존)
    log.info("  [Stage 4] MLP 채널 프루닝 (2SSP L2 norm 기준)")
    for li in tqdm(range(num_blocks), desc="Pruning MLP (2SSP+DLP)"):
        num_preserve = num_preserve_per_layer[li]
        _, top_indices = torch.topk(average_norms[li], num_preserve)
        mask = torch.ones_like(average_norms[li])
        mask[top_indices] = 0
        prune_mlp(model, mask, li)

    # config 갱신: per-layer intermediate_size + 호환용 intermediate_size
    model.config.intermediate_size = min(num_preserve_per_layer)
    if model.config.model_type == "qwen3":
        model.config.intermediate_size_per_layer = num_preserve_per_layer.copy()
        model.config.model_type = "qwen3_2ssp_dlp"
        model.config.auto_map = {
            "AutoConfig": "qwen3_2ssp_dlp.configuration_qwen3_2ssp_dlp.Qwen3Config2SSPDLP",
            "AutoModelForCausalLM": "qwen3_2ssp_dlp.modeling_qwen3_2ssp_dlp.Qwen3ForCausalLM2SSPDLP",
        }
    pruned_params = sum((mlp_hidden_size - n) * params_per_channel for n in num_preserve_per_layer)
    log.info(f"    MLP 채널 프루닝 완료 (제거 파라미터: {pruned_params/1e6:.2f}M)")

    if num_attn_submodules_to_prune > 0:
        log.info(f"  [Stage 5] Attention 서브모듈 제거 ({num_attn_submodules_to_prune}개)")
        from _2ssp_src.utilities import second_stage_attention, maskModel
        calibration_input_ids = torch.cat(calibration_dataset[:1], dim=1)
        attnMask, mlpMask = second_stage_attention(
            model, num_prune=num_attn_submodules_to_prune,
            calibration_input_ids=calibration_input_ids
        )
        maskModel(model, attnMask=attnMask, mlpMask=mlpMask)
        model.config.attention_pruned_layer_indices = [
            i for i in range(num_blocks) if attnMask[i] == 1
        ]
        log.info(f"    Attention 제거 완료 (제거 파라미터: {num_attn_submodules_to_prune * attn_total_params/1e6:.2f}M)")
    else:
        log.info("  [Stage 5] Attention 제거 생략 (prune_attention=0)")
        model.config.attention_pruned_layer_indices = []

    final_params = sum(p.numel() for p in model.parameters())
    log.info(f"  최종 파라미터: {final_params/1e6:.2f}M (원본 대비 {100*final_params/original_total_params:.1f}%)")

    return model


def get_calibration_from_dlp(dataloader, nsamples=32):
    """DLP dataloader를 2SSP calibration 형식으로 변환"""
    calibration = []
    for i, (inp, _) in enumerate(dataloader):
        if i >= nsamples:
            break
        calibration.append(inp.squeeze(0))
    return calibration

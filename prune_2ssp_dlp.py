"""
2SSP + DLP 결합 프루닝
- 채널 선택: 2SSP의 L2 norm (hidden state 기반) - 어떤 채널을 자를지
- 레이어별 sparsity: DLP의 layer-wise 비율 조정 - 각 레이어마다 다른 sparsity
"""

import sys
import os
import torch
import numpy as np
from tqdm import tqdm

# 2SSP 경로 추가 (root에서 실행 시)
_2ssp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '2SSP')
if _2ssp_path not in sys.path:
    sys.path.insert(0, _2ssp_path)
from src.utilities import get_mlp_hidden_state, prune_mlp


def get_mlp_hidden_state_batch(model, calibration_samples, device):
    """
    2SSP 스타일: 여러 calibration 샘플에 대해 MLP hidden state 수집
    calibration_samples: list of tensors (1, seq_len) each
    Returns: dict layer_idx -> list of tensors (seq_len, intermediate_size)
    """
    hidden_states_list = {}  # layer_idx -> list of tensors
    
    for sample in calibration_samples:
        if model.config.model_type in ("llama", "mistral", "phi3", "qwen2"):
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
            if model.config.model_type in ("llama", "mistral", "phi3", "qwen2"):
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
    ratio_conn = [1 - ls / total_conn for ls in layer_score]

    imp_ratios = torch.tensor(ratio_conn)
    min_ratio = torch.min(imp_ratios)
    max_ratio = torch.max(imp_ratios)
    if max_ratio - min_ratio < 1e-8:
        scaled_ratios = torch.ones_like(imp_ratios) * (1 - sparsity_ratio)
    else:
        scaled_ratios = (imp_ratios - min_ratio) * (1 / (max_ratio - min_ratio) * alpha_dlp * 2)
    all_layer_ratio = (scaled_ratios - torch.mean(scaled_ratios) + (1 - sparsity_ratio)).tolist()

    return all_layer_ratio, average_norms


def prune_mlp_2ssp_dlp(model, calibration_dataset, pruning_rate, alpha=1.5, alpha_dlp=0.15,
                        num_attn_submodules_to_prune=None):
    """
    2SSP + DLP 통합: 전체 목표 sparsity에서 Attention vs MLP 비율 조정 후 MLP 레이어별 프루닝

    - pruning_rate: 전체 목표 sparsity (0.5 = 50% 파라미터 제거)
    - alpha: Attention vs MLP 비율 조정 (2SSP Equation 5, default 1.5)
            alpha↑ → Attention 더 많이 제거, alpha↓ → MLP에 더 분배
    - alpha_dlp: 레이어별 sparsity 스케일링 (DLP)
    - num_attn_submodules_to_prune: None이면 alpha로 계산, 0이면 MLP만, >0이면 지정 개수 제거
    """
    layers = model.model.layers
    num_blocks = len(layers)
    mlp_hidden_size = model.config.intermediate_size

    # 파라미터 수
    main_model_total_params = sum(p.numel() for p in model.model.layers.parameters())
    attn_total_params = sum(p.numel() for p in model.model.layers[0].self_attn.parameters())
    mlp_total_params = sum(p.numel() for p in model.model.layers[0].mlp.parameters())

    # 1. Attention vs MLP 비율 (alpha로 조정)
    if num_attn_submodules_to_prune is None:
        num_attn_submodules_to_prune = round(
            num_blocks * pow(pruning_rate, (mlp_total_params / attn_total_params) / alpha)
        )
    num_attn_submodules_to_prune = max(0, min(num_attn_submodules_to_prune, num_blocks))

    # 2. 전체 목표에 맞춰 MLP에서 제거할 파라미터 수
    parameters_pruned_for_attention = num_attn_submodules_to_prune * attn_total_params
    target_parameters_to_prune = int(round(pruning_rate * main_model_total_params))
    total_mlp_params_to_prune = target_parameters_to_prune - parameters_pruned_for_attention
    total_mlp_params_to_prune = max(0, total_mlp_params_to_prune)

    # MLP 채널당 파라미터 (gate + up + down의 한 행/열)
    params_per_channel = 3 * model.config.hidden_size
    total_channels_to_prune = int(total_mlp_params_to_prune / params_per_channel)
    total_channels_to_prune = min(total_channels_to_prune, num_blocks * mlp_hidden_size - num_blocks)

    # 3. DLP 스타일 레이어별 비율 + 2SSP L2 norm 계산
    imp_ratios, average_norms = get_dlp_ratios_2ssp(
        model, calibration_dataset, total_channels_to_prune / (num_blocks * mlp_hidden_size), alpha_dlp
    )

    # 4. DLP 비율로 레이어별 채널 제거량 분배 (덜 중요한 레이어 = 더 많이 prune)
    prune_weights = [1 - r for r in imp_ratios]
    total_weight = sum(prune_weights)
    if total_weight < 1e-8:
        prune_weights = [1.0 / num_blocks] * num_blocks
        total_weight = 1.0
    num_prune_per_layer = [
        int(round(total_channels_to_prune * w / total_weight))
        for w in prune_weights
    ]
    # 정수 반올림 보정
    diff = total_channels_to_prune - sum(num_prune_per_layer)
    if diff > 0:
        for _ in range(diff):
            idx = np.argmax(num_prune_per_layer)
            if num_prune_per_layer[idx] < mlp_hidden_size - 1:
                num_prune_per_layer[idx] += 1
    elif diff < 0:
        for _ in range(-diff):
            idx = np.argmin(num_prune_per_layer)
            if num_prune_per_layer[idx] > 0:
                num_prune_per_layer[idx] -= 1

    # 5. 각 레이어별 보존할 채널 수
    num_prune_per_layer = [min(n, mlp_hidden_size - 1) for n in num_prune_per_layer]
    num_preserve_per_layer = [
        max(1, mlp_hidden_size - n)
        for n in num_prune_per_layer
    ]

    # 6. 2SSP 기준으로 채널 선택 (L2 norm 상위 채널 보존)
    for li in tqdm(range(num_blocks), desc="Pruning MLP (2SSP+DLP)"):
        num_preserve = num_preserve_per_layer[li]
        _, top_indices = torch.topk(average_norms[li], num_preserve)
        mask = torch.ones_like(average_norms[li])
        mask[top_indices] = 0
        prune_mlp(model, mask, li)

    model.config.intermediate_size = min(num_preserve_per_layer)

    if num_attn_submodules_to_prune > 0:
        from src.utilities import second_stage_attention, maskModel  # noqa: F811
        calibration_input_ids = torch.cat(calibration_dataset[:1], dim=1)
        attnMask, mlpMask = second_stage_attention(
            model, num_prune=num_attn_submodules_to_prune,
            calibration_input_ids=calibration_input_ids
        )
        maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

    return model


def get_calibration_from_dlp(dataloader, nsamples=32):
    """DLP dataloader를 2SSP calibration 형식으로 변환"""
    calibration = []
    for i, (inp, _) in enumerate(dataloader):
        if i >= nsamples:
            break
        calibration.append(inp.squeeze(0))
    return calibration

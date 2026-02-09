"""2SSP utilities - embedded for 2SSP+DLP integration"""
import logging
import torch

log = logging.getLogger(__name__)
from types import MethodType
from tqdm import tqdm
from .evaluation import evaluate_perplexity


def maskModel(model, attnMask, mlpMask):
    for i in range(len(attnMask)):
        if attnMask[i] == 1 and mlpMask[i] == 1:
            def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
                return (hidden_states,)
            model.model.layers[i].forward_bak = model.model.layers[i].forward
            model.model.layers[i].forward = MethodType(identity_forward, model.model.layers[i])
        elif attnMask[i] == 1 and mlpMask[i] == 0:
            if model.config.model_type in ("phi", "phi3"):
                def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
                    return torch.zeros_like(hidden_states), None, None
            else:
                # LLaMA/Mistral/Qwen2/Qwen3: self_attn returns (attn_output, attn_weights) - 2 values
                def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
                    return torch.zeros_like(hidden_states), None
            model.model.layers[i].self_attn.forward_bak = model.model.layers[i].self_attn.forward
            model.model.layers[i].self_attn.forward = MethodType(identity_forward, model.model.layers[i].self_attn)
        elif attnMask[i] == 0 and mlpMask[i] == 1:
            if model.config.model_type in ("phi", "phi3"):
                def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
                    return torch.zeros_like(hidden_states)
            else:
                def identity_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
                    return 0
            model.model.layers[i].mlp.forward_bak = model.model.layers[i].mlp.forward
            model.model.layers[i].mlp.forward = MethodType(identity_forward, model.model.layers[i].mlp)


def unmaskModel(model, attnMask, mlpMask):
    for i in range(len(attnMask)):
        if attnMask[i] == 1 and mlpMask[i] == 1:
            model.model.layers[i].forward = model.model.layers[i].forward_bak
        elif attnMask[i] == 1 and mlpMask[i] == 0:
            model.model.layers[i].self_attn.forward = model.model.layers[i].self_attn.forward_bak
        elif attnMask[i] == 0 and mlpMask[i] == 1:
            model.model.layers[i].mlp.forward = model.model.layers[i].mlp.forward_bak


@torch.no_grad()
def get_mlp_hidden_state(model, calibration_sample):
    if model.config.model_type in ("llama", "mistral", "phi3", "qwen2", "qwen3", "qwen3_2ssp_dlp"):
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
        last_linear = layer.mlp.down_proj if model.config.model_type in ("llama", "mistral", "phi3", "qwen2", "qwen3", "qwen3_2ssp_dlp") else layer.mlp.fc2
        hooks.append(last_linear.register_forward_hook(lambda m, inp, out: hook(m, inp, out)))

    input_ids = calibration_sample.to(model.device)
    with torch.no_grad():
        _ = model(input_ids)

    for h in hooks:
        h.remove()

    return hidden_states


@torch.no_grad()
def prune_mlp(model, mask, block_i):
    preserve_mask = torch.where(mask == 0)[0]
    new_intermediate_size = preserve_mask.size(0)
    layer = model.model.layers[block_i]

    if model.config.model_type in ("llama", "mistral", "qwen2", "qwen3", "qwen3_2ssp_dlp"):
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[preserve_mask]
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[preserve_mask]
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, preserve_mask]
        layer.mlp.gate_proj.weight.out_features = new_intermediate_size
        layer.mlp.gate_proj.out_features = new_intermediate_size
        layer.mlp.up_proj.weight.out_features = new_intermediate_size
        layer.mlp.up_proj.out_features = new_intermediate_size
        layer.mlp.down_proj.weight.in_features = new_intermediate_size
        layer.mlp.down_proj.in_features = new_intermediate_size
    elif model.config.model_type == "phi3":
        gate_up_weights = layer.mlp.gate_up_proj.weight.data
        gate_weights, up_weights = gate_up_weights.chunk(2, dim=0)
        gate_weights = gate_weights[preserve_mask]
        up_weights = up_weights[preserve_mask]
        layer.mlp.gate_up_proj.weight.data = torch.cat([gate_weights, up_weights], dim=0)
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, preserve_mask]
    elif model.config.model_type == "phi":
        layer.mlp.fc1.weight.data = layer.mlp.fc1.weight.data[preserve_mask]
        layer.mlp.fc1.bias.data = layer.mlp.fc1.bias.data[preserve_mask]
        layer.mlp.fc2.weight.data = layer.mlp.fc2.weight.data[:, preserve_mask]
    else:
        raise ValueError(f"Unsupported model_type: {model.config.model_type}")


@torch.no_grad()
def second_stage_attention(model, num_prune, calibration_input_ids):
    num_blocks = len(model.model.layers)
    attnMask = [0] * num_blocks
    mlpMask = [0] * num_blocks

    ppl = evaluate_perplexity(model, calibration_input_ids, seq_len=2048, enable_tqdm=False)
    log.info(f"    Original perplexity: {ppl:.4f}")

    for step in tqdm(range(num_prune), desc="    Attention pruning"):
        best_to_prune = None
        best_ppl = float("inf")

        for to_prune in range(num_blocks):
            if attnMask[to_prune] == 1:
                continue
            attnMask[to_prune] = 1
            maskModel(model, attnMask=attnMask, mlpMask=mlpMask)
            ppl = evaluate_perplexity(model, calibration_input_ids, seq_len=2048, enable_tqdm=False)
            log.debug(f"[Attention] When pruning {to_prune} perplexity is {ppl}")
            if ppl < best_ppl:
                best_ppl = ppl
                best_to_prune = to_prune
            unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)
            attnMask[to_prune] = 0

        log.info(f"      prune #{step+1}: layer {best_to_prune} 제거 (ppl={best_ppl:.4f})")
        attnMask[best_to_prune] = 1

        if model.config.model_type in ("llama", "mistral", "qwen2", "qwen3", "qwen3_2ssp_dlp"):
            del model.model.layers[best_to_prune].self_attn.q_proj
            del model.model.layers[best_to_prune].self_attn.k_proj
            del model.model.layers[best_to_prune].self_attn.v_proj
            del model.model.layers[best_to_prune].self_attn.o_proj
        elif model.config.model_type == "phi3":
            del model.model.layers[best_to_prune].self_attn.qkv_proj
            del model.model.layers[best_to_prune].self_attn.o_proj
        elif model.config.model_type == "phi":
            del model.model.layers[best_to_prune].self_attn.q_proj
            del model.model.layers[best_to_prune].self_attn.k_proj
            del model.model.layers[best_to_prune].self_attn.v_proj
            del model.model.layers[best_to_prune].self_attn.dense
        else:
            raise ValueError(f"Unsupported model_type: {model.config.model_type}")

    return attnMask, mlpMask

"""
Qwen3 커스텀 모델: per-layer intermediate_size 지원 (2SSP+DLP 저장/로딩용)
"""
from typing import Optional

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from .configuration_qwen3_2ssp_dlp import Qwen3Config2SSPDLP


class Qwen3MLP2SSPDLP(nn.Module):
    """Qwen3MLP with per-layer intermediate_size"""

    def __init__(self, config, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else config.intermediate_size
        )
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )


class Qwen3DecoderLayer2SSPDLP(Qwen3DecoderLayer):
    """Qwen3DecoderLayer with per-layer MLP intermediate_size"""

    def __init__(self, config: Qwen3Config2SSPDLP, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        if (
            hasattr(config, "intermediate_size_per_layer")
            and config.intermediate_size_per_layer is not None
        ):
            inter_size = config.intermediate_size_per_layer[layer_idx]
        else:
            inter_size = None
        self.mlp = Qwen3MLP2SSPDLP(config, intermediate_size=inter_size)

        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]


class Qwen3Model2SSPDLP(Qwen3Model):
    """Qwen3Model with per-layer intermediate_size"""

    def __init__(self, config: Qwen3Config2SSPDLP):
        super().__init__(config)
        # Replace layers with per-layer intermediate_size
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer2SSPDLP(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )


class Qwen3ForCausalLM2SSPDLP(Qwen3ForCausalLM):
    """Qwen3ForCausalLM with per-layer intermediate_size (2SSP+DLP)"""

    config_class = Qwen3Config2SSPDLP

    def __init__(self, config: Qwen3Config2SSPDLP):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3Model2SSPDLP(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.post_init()

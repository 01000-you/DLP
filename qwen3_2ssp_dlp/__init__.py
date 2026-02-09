"""
Qwen3 2SSP+DLP: per-layer intermediate_size 저장/로딩 지원
"""
from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_qwen3_2ssp_dlp import Qwen3Config2SSPDLP
from .modeling_qwen3_2ssp_dlp import Qwen3ForCausalLM2SSPDLP

__all__ = ["Qwen3Config2SSPDLP", "Qwen3ForCausalLM2SSPDLP", "register_qwen3_2ssp_dlp"]


def register_qwen3_2ssp_dlp():
    """HuggingFace Auto 클래스에 등록 (저장된 qwen3_2ssp_dlp 모델 로딩용)"""
    try:
        AutoConfig.register("qwen3_2ssp_dlp", Qwen3Config2SSPDLP)
        AutoModelForCausalLM.register(
            Qwen3Config2SSPDLP, Qwen3ForCausalLM2SSPDLP
        )
    except ValueError:
        pass  # 이미 등록됨

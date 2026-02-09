"""
Qwen3Config 확장: per-layer intermediate_size 지원 (2SSP+DLP)
"""
from typing import List, Optional

from transformers import Qwen3Config


class Qwen3Config2SSPDLP(Qwen3Config):
    """Qwen3Config + intermediate_size_per_layer (2SSP+DLP 저장/로딩용)"""

    model_type = "qwen3_2ssp_dlp"

    def __init__(
        self,
        intermediate_size_per_layer: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_size_per_layer = intermediate_size_per_layer

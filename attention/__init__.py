"""Extracao de mapas de atencao em VED e VLM."""

from codigo.attention.vlm_extractor import (
    VLMAttentionResult,
    extract_vlm_attention,
    inspect_vlm_architecture,
)

__all__ = [
    "VLMAttentionResult",
    "extract_vlm_attention",
    "inspect_vlm_architecture",
]


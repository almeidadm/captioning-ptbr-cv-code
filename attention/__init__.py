"""Extracao e analise de mapas de atencao em VED e VLM."""

from codigo.attention.auto_classify import (
    AttentionMetrics,
    Category,
    EDGE_REGIONS,
    REGION_NAMES,
    compute_attention_metrics,
    is_edge_region,
    peak_to_region,
    regions_match,
    render_suggestion_table_markdown,
    suggest_category,
    suggest_for_word_table,
)
from codigo.attention.vlm_extractor import (
    VLMAttentionResult,
    extract_vlm_attention,
    inspect_vlm_architecture,
)

__all__ = [
    "AttentionMetrics",
    "Category",
    "EDGE_REGIONS",
    "REGION_NAMES",
    "VLMAttentionResult",
    "compute_attention_metrics",
    "extract_vlm_attention",
    "inspect_vlm_architecture",
    "is_edge_region",
    "peak_to_region",
    "regions_match",
    "render_suggestion_table_markdown",
    "suggest_category",
    "suggest_for_word_table",
]


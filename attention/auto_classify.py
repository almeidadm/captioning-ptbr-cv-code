"""Sugestao automatica de categoria para o protocolo de inspecao.

Modulo de apoio ao sanity check qualitativo
(`notas/metodologia/protocolo-inspecao-heatmap.md`,
`notas/metodologia/aplicacao-simetrica-protocolo-ved-vlm.md`).

A funcao `suggest_category` retorna uma sugestao preliminar de
categoria com base em:
- tipo da palavra (stopword, substantivo consensual/isolado/nao-coberto, especial)
- metricas do mapa de atencao (entropia, top-k mass, centro de massa)
- regiao anotada na Etapa 1 do protocolo (quando o tipo e substantivo)

A saida e **sugestao**, nao decisao. O revisor confirma ou corrige.
Em S3, calibrar empiricamente os thresholds em S2 antes de fixar.

Categorias retornadas (alinhadas com o protocolo):
    "PLAUSIVEL", "DIFUSO", "DESLOCADO", "ARTEFATO",
    "HALLUCINADO-PLAUSIVEL", "HALLUCINADO-IRREAL", "N/A", None

`None` significa "indecidivel automaticamente — revisor decide".
HALLUCINADO-PLAUSIVEL nao pode ser sugerido automaticamente: requer
julgamento visual humano sobre se o objeto realmente esta na imagem
(o auto-classify nao acessa a imagem). Sugere-se sempre HALLUCINADO-IRREAL
para nao-cobertos com atencao em regiao vazia, e None caso contrario.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# Vocabulario fixo de 9 regioes do protocolo
REGION_NAMES = [
    "topo-esquerdo", "topo-centro", "topo-direito",
    "meio-esquerdo", "centro", "meio-direito",
    "inferior-esquerdo", "inferior-centro", "inferior-direito",
]

# Regioes consideradas "borda" para fins de ARTEFATO
EDGE_REGIONS = frozenset({
    "topo-esquerdo", "topo-direito",
    "inferior-esquerdo", "inferior-direito",
})


Category = Literal[
    "PLAUSIVEL", "DIFUSO", "DESLOCADO", "ARTEFATO",
    "HALLUCINADO-PLAUSIVEL", "HALLUCINADO-IRREAL", "N/A",
]


@dataclass(frozen=True)
class AttentionMetrics:
    """Metricas de um mapa de atencao 2D."""

    entropy_norm: float       # entropia normalizada em [0, 1]
    top1_mass: float          # massa do patch mais ativado, em [0, 1]
    top5_mass: float          # massa dos 5 patches mais ativados
    peak_region: str          # uma das 9 regioes
    peak_in_edge: bool        # se peak_region esta em EDGE_REGIONS
    com_h: float              # centro de massa (linha) em [0, 1]
    com_w: float              # centro de massa (coluna) em [0, 1]


def compute_attention_metrics(attn_word: np.ndarray) -> AttentionMetrics:
    """Calcula metricas escalares de um mapa de atencao 2D.

    Args:
        attn_word: array (H, W) com pesos de atencao. Nao precisa estar
            normalizado; sera renormalizado internamente.

    Returns:
        AttentionMetrics com todas as metricas computadas.
    """
    if attn_word.ndim != 2:
        raise ValueError(f"attn_word deve ser 2D (H, W), got shape {attn_word.shape}")

    H, W = attn_word.shape
    flat = attn_word.flatten().astype(np.float64)

    # Normaliza para distribuicao de probabilidade
    total = flat.sum()
    if total <= 0:
        # Mapa vazio: retornar metricas degeneradas
        return AttentionMetrics(
            entropy_norm=1.0, top1_mass=0.0, top5_mass=0.0,
            peak_region="centro", peak_in_edge=False,
            com_h=0.5, com_w=0.5,
        )
    p = flat / total

    # Entropia normalizada
    eps = 1e-12
    entropy = -np.sum(p * np.log(p + eps))
    max_entropy = np.log(H * W)
    entropy_norm = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    # Top-k mass
    sorted_p = np.sort(p)[::-1]
    top1_mass = float(sorted_p[0])
    top5_mass = float(sorted_p[: min(5, len(sorted_p))].sum())

    # Pico
    peak_idx = int(p.argmax())
    peak_h, peak_w = peak_idx // W, peak_idx % W
    peak_region = peak_to_region(peak_h, peak_w, H, W)
    peak_in_edge = peak_region in EDGE_REGIONS

    # Centro de massa (normalizado em [0, 1])
    coords_h = np.arange(H).reshape(-1, 1)
    coords_w = np.arange(W).reshape(1, -1)
    grid_p = p.reshape(H, W)
    com_h = float((grid_p * coords_h).sum() / max(H - 1, 1))
    com_w = float((grid_p * coords_w).sum() / max(W - 1, 1))

    return AttentionMetrics(
        entropy_norm=entropy_norm,
        top1_mass=top1_mass,
        top5_mass=top5_mass,
        peak_region=peak_region,
        peak_in_edge=peak_in_edge,
        com_h=com_h,
        com_w=com_w,
    )


def peak_to_region(peak_h: int, peak_w: int, H: int, W: int) -> str:
    """Mapeia coordenadas (peak_h, peak_w) para uma das 9 regioes 3x3.

    Args:
        peak_h: coordenada vertical do pico em [0, H-1].
        peak_w: coordenada horizontal do pico em [0, W-1].
        H: altura do grid.
        W: largura do grid.

    Returns:
        String com nome da regiao (uma das `REGION_NAMES`).
    """
    row = min(2, int(peak_h * 3 / H))
    col = min(2, int(peak_w * 3 / W))
    return REGION_NAMES[row * 3 + col]


def is_edge_region(region: str) -> bool:
    """True se a regiao e canto (parte de EDGE_REGIONS)."""
    return region in EDGE_REGIONS


def regions_match(observed: str, expected: str | set[str]) -> bool:
    """Verifica se a regiao observada bate com a expectativa.

    Aceita expected como string unica ou set de regioes (uniao).
    Match exato — sem fuzzy.
    """
    if isinstance(expected, str):
        # Expected pode ser composto por "+", e.g. "centro+meio-direito"
        expected_set = {r.strip() for r in expected.split("+")}
    else:
        expected_set = expected
    return observed in expected_set


def suggest_category(
    word_text: str,
    word_type: str,
    metrics: AttentionMetrics,
    region_anotada: str | set[str] | None = None,
    *,
    diffuse_entropy_threshold: float = 0.85,
    focused_entropy_threshold: float = 0.65,
    artefact_top1_threshold: float = 0.10,
) -> Category | None:
    """Sugere categoria automatica para uma palavra.

    A logica segue o protocolo. Casos com sinal claro recebem sugestao;
    casos ambiguos retornam None.

    Args:
        word_text: palavra reconstruida.
        word_type: categoria automatica preliminar do tipo (saida de
            `classify_word_type`). Valores esperados: "especial",
            "stopword", "substantivo-cons", "substantivo-iso",
            "substantivo-nao-coberto", "outro".
        metrics: AttentionMetrics computado em compute_attention_metrics.
        region_anotada: regiao(oes) anotada(s) na Etapa 1 do protocolo
            para o substantivo. Pode ser string unica ("centro"),
            string composta ("centro+meio-direito"), ou set. Se None
            (nao anotado), so tipos sem dependencia de regiao podem
            ser sugeridos.
        diffuse_entropy_threshold: acima disso, atencao e considerada
            difusa (default 0.85; calibrar em S2).
        focused_entropy_threshold: abaixo disso, atencao e considerada
            focada o suficiente para classificar PLAUSIVEL/DESLOCADO
            (default 0.65; calibrar em S2).
        artefact_top1_threshold: pico unico com mais que isso de massa
            em regiao de borda sem objeto e candidato a ARTEFATO
            (default 0.10).

    Returns:
        Sugestao de categoria, ou None se indecidivel automaticamente.
    """
    # Especiais -> N/A direto
    if word_type == "especial":
        return "N/A"

    # Stopwords: so DIFUSO ou ARTEFATO permitidos
    if word_type == "stopword":
        if metrics.peak_in_edge and metrics.top1_mass >= artefact_top1_threshold:
            return "ARTEFATO"
        if metrics.entropy_norm >= diffuse_entropy_threshold:
            return "DIFUSO"
        # Stopword com atencao focada no meio sem ser artefato — incomum
        # mas nao classificavel automaticamente
        return None

    # Substantivos consensuais ou isolados: precisam de regiao anotada
    if word_type in ("substantivo-cons", "substantivo-iso"):
        if region_anotada is None:
            # Sem gabarito da Etapa 1, nao da pra classificar PLAUSIVEL/DESLOCADO
            return None
        if metrics.peak_in_edge and metrics.top1_mass >= artefact_top1_threshold:
            return "ARTEFATO"
        if metrics.entropy_norm >= diffuse_entropy_threshold:
            return "DIFUSO"
        if metrics.entropy_norm <= focused_entropy_threshold:
            if regions_match(metrics.peak_region, region_anotada):
                return "PLAUSIVEL"
            return "DESLOCADO"
        # Zona intermediaria: deixar para revisor
        return None

    # Substantivos nao cobertos: HALLUCINADO-IRREAL ou None
    if word_type == "substantivo-nao-coberto":
        if metrics.peak_in_edge and metrics.top1_mass >= artefact_top1_threshold:
            return "ARTEFATO"
        if metrics.entropy_norm >= diffuse_entropy_threshold:
            return "DIFUSO"
        # Para distinguir HALLUCINADO-PLAUSIVEL vs IRREAL precisa
        # confirmar visualmente se o objeto existe na imagem.
        # Auto-classify nao tem acesso a essa info; revisor decide.
        return None

    # "outro" (verbos, adjetivos, etc.): sem regra simples
    return None


def suggest_for_word_table(
    rows: list[dict],
    attn_grid: np.ndarray,
    regions_anotadas: dict[str, str | set[str]] | None = None,
    aggregate_strategy: str = "mean",
    **suggest_kwargs,
) -> list[dict]:
    """Aplica suggest_category a todas as linhas da tabela de palavras.

    Args:
        rows: saida de `build_word_table` (lista de dicts).
        attn_grid: array (T, H, W) original.
        regions_anotadas: dict mapeando texto-da-palavra (lowercase)
            para regiao anotada. None se nenhuma regiao foi anotada
            ainda.
        aggregate_strategy: agregacao da atencao por palavra (passada
            a `aggregate_attention`).
        **suggest_kwargs: thresholds passados a `suggest_category`.

    Returns:
        Mesmas rows com campos novos:
            metrics: AttentionMetrics
            categoria_sugerida: Category | None
    """
    from codigo.utils.word_reconstruction import (
        ReconstructedWord, aggregate_attention,
    )

    out_rows = []
    for row in rows:
        word_obj = ReconstructedWord(
            text=row["text"],
            subword_indices=row["subword_indices"],
            subword_tokens=row["subwords"],
        )
        attn_word = aggregate_attention(attn_grid, word_obj, strategy=aggregate_strategy)
        metrics = compute_attention_metrics(attn_word)

        text_lower = row["text"].lower().strip()
        region = (
            regions_anotadas.get(text_lower)
            if regions_anotadas is not None else None
        )

        suggestion = suggest_category(
            word_text=row["text"],
            word_type=row["tipo_sugerido"],
            metrics=metrics,
            region_anotada=region,
            **suggest_kwargs,
        )

        out_rows.append({
            **row,
            "metrics": metrics,
            "categoria_sugerida": suggestion,
        })
    return out_rows


def render_suggestion_table_markdown(rows: list[dict]) -> str:
    """Renderiza a tabela com sugestoes como markdown."""
    lines = [
        "| w | Palavra | Tipo | n_refs | Pico | Entropia | Top1 | Sugestao | drift |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        m: AttentionMetrics = row["metrics"]
        n_refs = row["n_refs"]
        n_refs_str = "-" if n_refs == -1 else str(n_refs)
        suggestion = row["categoria_sugerida"] or "?"
        drift = "sim" if row.get("drift_alto") else (
            "-" if len(row["subword_indices"]) == 1 else "nao"
        )
        lines.append(
            f"| {row['w']} | {row['text']} | {row['tipo_sugerido']} | "
            f"{n_refs_str} | {m.peak_region} | {m.entropy_norm:.2f} | "
            f"{m.top1_mass:.2f} | {suggestion} | {drift} |"
        )
    return "\n".join(lines)

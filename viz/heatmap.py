"""Sobreposicao de heatmap de atencao em imagem PIL."""

from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_heatmap(
    image: Image.Image,
    attn_grid: np.ndarray,
    *,
    alpha: float = 0.5,
    cmap: str = "jet",
) -> Image.Image:
    """Sobrepoe um mapa de atencao 2D a uma imagem RGB.

    Args:
        image: imagem PIL em RGB.
        attn_grid: array 2D (H_grid, W_grid) com valores >= 0.
        alpha: peso do heatmap na composicao (0 = so imagem, 1 = so heatmap).
        cmap: nome do colormap matplotlib.

    Returns:
        Imagem PIL com heatmap sobreposto, no tamanho original da imagem.
    """
    import matplotlib.cm as cm

    if attn_grid.ndim != 2:
        raise ValueError(f"attn_grid deve ser 2D, recebido shape {attn_grid.shape}")

    img_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    grid_norm = _normalize(attn_grid)
    heatmap_grid = Image.fromarray((grid_norm * 255).astype(np.uint8))
    heatmap_resized = heatmap_grid.resize(image.size, resample=Image.BILINEAR)
    heatmap_arr = np.asarray(heatmap_resized, dtype=np.float32) / 255.0

    colormap = cm.get_cmap(cmap)
    heatmap_rgb = colormap(heatmap_arr)[..., :3].astype(np.float32)

    blended = (1 - alpha) * img_rgb + alpha * heatmap_rgb
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def plot_attention_grid(
    image: Image.Image,
    attn_per_token: np.ndarray,
    tokens: list[str],
    *,
    n_cols: int = 4,
    figsize_per_cell: tuple[float, float] = (3.0, 3.0),
):
    """Plota um grid (token x heatmap) para inspecao qualitativa.

    Args:
        image: imagem PIL.
        attn_per_token: array (T, H_grid, W_grid).
        tokens: lista de T strings descrevendo cada passo.
        n_cols: colunas do grid de subplots.
        figsize_per_cell: tamanho de cada subplot.

    Returns:
        Figure matplotlib (caller decide se salva ou mostra).
    """
    import matplotlib.pyplot as plt

    if attn_per_token.shape[0] != len(tokens):
        raise ValueError(
            f"Numero de mapas ({attn_per_token.shape[0]}) != numero de tokens ({len(tokens)})"
        )

    n_tokens = len(tokens)
    n_rows = int(np.ceil(n_tokens / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_cell[0], n_rows * figsize_per_cell[1]),
    )
    axes = np.array(axes).reshape(-1)

    for i in range(n_rows * n_cols):
        ax = axes[i]
        ax.axis("off")
        if i >= n_tokens:
            continue
        overlay = overlay_heatmap(image, attn_per_token[i])
        ax.imshow(overlay)
        ax.set_title(tokens[i], fontsize=10)

    fig.tight_layout()
    return fig


def _normalize(x: np.ndarray) -> np.ndarray:
    """Normaliza para [0, 1] com guarda contra divisao por zero."""
    x_min, x_max = float(x.min()), float(x.max())
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)

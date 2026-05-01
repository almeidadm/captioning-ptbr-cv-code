"""Carrega imagens individuais para a PoC de S1.

Em S1 trabalhamos com 1 imagem por vez. O loader unifica acesso ao
Hugging Face Hub (datasets `laicsiifes/flickr30k-pt-br` e
`laicsiifes/flickr30k-pt-br-human-generated`) e a arquivos locais.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image


DatasetName = Literal[
    "laicsiifes/flickr30k-pt-br",
    "laicsiifes/flickr30k-pt-br-human-generated",
]


@dataclass(frozen=True)
class CaptionSample:
    """Unidade minima do pipeline: imagem + referencias + identificador."""

    image: Image.Image
    captions: list[str]
    image_id: str
    source: str


def load_sample_from_hf(
    dataset: DatasetName,
    *,
    split: str = "test",
    index: int = 0,
    cache_dir: str | Path | None = None,
) -> CaptionSample:
    """Carrega 1 amostra do split indicado.

    Args:
        dataset: nome do dataset no HF Hub.
        split: split a usar (`train` | `validation` | `test`).
        index: indice da amostra dentro do split.
        cache_dir: diretorio de cache (em Colab apontar para Drive).

    Returns:
        CaptionSample com imagem PIL, lista de captions e id da imagem.
    """
    from datasets import load_dataset

    ds = load_dataset(
        dataset,
        split=f"{split}[{index}:{index + 1}]",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    row = ds[0]
    image = row["image"].convert("RGB")
    captions = row.get("caption") or row.get("captions") or []
    if isinstance(captions, str):
        captions = [captions]
    image_id = str(row.get("filename") or row.get("img_id") or row.get("image_id") or index)
    return CaptionSample(
        image=image,
        captions=list(captions),
        image_id=image_id,
        source=dataset,
    )


def load_image_from_path(path: str | Path) -> Image.Image:
    """Carrega imagem do disco em RGB."""
    return Image.open(Path(path)).convert("RGB")

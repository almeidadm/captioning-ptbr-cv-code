"""Extracao de cross-attention em modelos Vision Encoder-Decoder.

Stub para S1. A implementacao completa entra em S2 quando o pipeline
escala para 1000 imagens. Em S1 a logica vive inline no notebook
para facilitar inspecao.
"""

from __future__ import annotations

# TODO(S2): mover para ca a logica de output.cross_attentions ->
#           tensor (T_gen, num_patches) -> reshape para grid espacial.

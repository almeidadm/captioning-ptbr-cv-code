"""Extracao de mapas de atencao visual em VLMs tipo LLaVA / TinyLLaVA / ViTucano.

Premissa arquitetural
---------------------
ViTucano = encoder visual (SigLIP/CLIP) + projetor MLP + LLM Tucano.
No `input_ids` existe **um** placeholder de imagem (`image_token_id`).
Durante o forward do LLM, esse placeholder e EXPANDIDO em N embeddings
visuais contiguos. O LLM faz self-attention causal sobre a sequencia
expandida, atendendo aos N tokens visuais como atendaria a tokens de texto.

Para cada token gerado, a atencao "visual" e a fatia da matriz de
atencao do LLM sobre a faixa de N posicoes correspondentes a imagem.

Estrategia de extracao
----------------------
1. Localizar a posicao `pos` do placeholder em `input_ids`.
2. Rodar `model.generate(..., output_attentions=True, return_dict_in_generate=True)`.
3. Inferir N a partir do `kv_len` da primeira camada de atencao do passo 0:
       N = kv_len_0 - (prompt_len - 1)
   (porque 1 placeholder vira N tokens, entao a sequencia expandida tem
    prompt_len - 1 + N tokens.)
4. Para cada passo t, agregar camadas e cabecas, pegar a ultima query
   (a que prediz o token gerado) e fatiar [pos, pos+N) nas keys.
5. Reshape para o grid espacial (H, W) com H*W = N.

Limites conhecidos
------------------
- Suporta 1 imagem por inferencia.
- Suporta apenas geracao greedy (num_beams=1) — beam search complica a
  associacao step -> token gerado.
- Assume que o vision encoder produz tokens em ordem row-major (padrao
  ViT/SigLIP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image


LayerAggregation = Literal["last", "mean", "first"]
HeadAggregation = Literal["mean", "max"]


@dataclass
class VLMAttentionResult:
    """Saida da extracao: caption + mapas de atencao por token."""

    caption: str
    generated_tokens: list[str]
    attn_grid: np.ndarray
    visual_token_range: tuple[int, int]
    grid_shape: tuple[int, int]
    n_visual_tokens: int
    notes: list[str] = field(default_factory=list)
    raw_attn_per_step: list[np.ndarray] | None = None


def inspect_vlm_architecture(model, processor) -> dict[str, Any]:
    """Coleta metadados arquiteturais para diagnostico.

    Util como primeira chamada apos carregar o modelo: imprime e retorna
    informacoes que ajudam a entender se a PoC vai funcionar (image token
    encontrado, vision tower presente, etc.).
    """
    info: dict[str, Any] = {}
    info["model_class"] = type(model).__name__
    info["torch_dtype"] = str(getattr(model, "dtype", "?"))

    image_token_id = (
        getattr(model.config, "image_token_id", None)
        or getattr(model.config, "image_token_index", None)
    )
    info["image_token_id"] = image_token_id

    tokenizer = getattr(processor, "tokenizer", processor)
    info["tokenizer_class"] = type(tokenizer).__name__
    if hasattr(tokenizer, "image_token"):
        info["processor_image_token"] = getattr(tokenizer, "image_token", None)

    vision_tower = (
        getattr(model, "vision_tower", None)
        or getattr(model, "vision_model", None)
        or getattr(getattr(model, "model", None), "vision_tower", None)
    )
    if vision_tower is not None:
        info["vision_tower_class"] = type(vision_tower).__name__
        v_cfg = getattr(vision_tower, "config", None)
        if v_cfg is not None:
            info["vision_image_size"] = getattr(v_cfg, "image_size", None)
            info["vision_patch_size"] = getattr(v_cfg, "patch_size", None)
            info["vision_hidden_size"] = getattr(v_cfg, "hidden_size", None)

    llm = (
        getattr(model, "language_model", None)
        or getattr(getattr(model, "model", None), "language_model", None)
    )
    if llm is not None and hasattr(llm, "config"):
        info["llm_class"] = type(llm).__name__
        info["llm_n_layers"] = getattr(llm.config, "num_hidden_layers", None)
        info["llm_n_heads"] = getattr(llm.config, "num_attention_heads", None)
        info["llm_hidden_size"] = getattr(llm.config, "hidden_size", None)

    return info


def _estimate_grid_shape(n_visual: int) -> tuple[int, int]:
    """Infere grid (H, W) com H*W = n_visual ou H*W = n_visual - 1 (caso CLS).

    Casos comuns para ViT/SigLIP: 196 (14x14), 256 (16x16), 576 (24x24), 729 (27x27).
    """
    h = int(round(math.sqrt(n_visual)))
    if h * h == n_visual:
        return (h, h)
    h = int(round(math.sqrt(n_visual - 1)))
    if h * h == n_visual - 1:
        return (h, h)
    raise ValueError(
        f"Nao foi possivel inferir grid quadrado para n_visual={n_visual}; "
        "passar grid_shape explicito."
    )


def _resolve_image_token_id(model, override: int | None) -> int:
    if override is not None:
        return override
    candidates = [
        getattr(model.config, "image_token_id", None),
        getattr(model.config, "image_token_index", None),
        getattr(model.config, "image_index", None),
    ]
    for c in candidates:
        if c is not None:
            return int(c)
    raise ValueError(
        "image_token_id nao encontrado em model.config "
        "(image_token_id / image_token_index / image_index). "
        "Passar explicitamente via argumento."
    )


def _find_placeholder_position(input_ids: torch.Tensor, image_token_id: int) -> int:
    ids = input_ids[0].tolist()
    positions = [i for i, t in enumerate(ids) if t == image_token_id]
    if len(positions) == 0:
        raise ValueError(
            f"Placeholder de imagem (id={image_token_id}) nao encontrado em input_ids. "
            "O processor pode nao ter inserido o token — verificar prompt e processor."
        )
    if len(positions) > 1:
        raise ValueError(
            f"Multiplos placeholders de imagem encontrados ({len(positions)}). "
            "PoC suporta apenas 1 imagem por inferencia."
        )
    return positions[0]


def _aggregate_layers(step_attentions: tuple, mode: LayerAggregation) -> torch.Tensor:
    if mode == "last":
        return step_attentions[-1]
    if mode == "first":
        return step_attentions[0]
    if mode == "mean":
        return torch.stack(list(step_attentions), dim=0).mean(dim=0)
    raise ValueError(f"layer_aggregation invalido: {mode}")


def _aggregate_heads(layer_attn: torch.Tensor, mode: HeadAggregation) -> torch.Tensor:
    if mode == "max":
        return layer_attn.amax(dim=1)
    if mode == "mean":
        return layer_attn.mean(dim=1)
    raise ValueError(f"head_aggregation invalido: {mode}")


@torch.no_grad()
def extract_vlm_attention(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int = 30,
    layer_aggregation: LayerAggregation = "last",
    head_aggregation: HeadAggregation = "mean",
    image_token_id: int | None = None,
    grid_shape: tuple[int, int] | None = None,
    device: str = "cuda",
    return_raw: bool = False,
) -> VLMAttentionResult:
    """Gera caption e extrai mapas de atencao visual por token gerado.

    Args:
        model: VLM tipo LLaVA / TinyLLaVA / ViTucano carregado em `eval`.
        processor: AutoProcessor correspondente.
        image: PIL.Image em RGB.
        prompt: texto do usuario (ex: "Descreva esta imagem em portugues...").
        max_new_tokens: comprimento maximo da geracao.
        layer_aggregation: 'last' | 'first' | 'mean' sobre as camadas do LLM.
        head_aggregation: 'mean' | 'max' sobre as cabecas de atencao.
        image_token_id: id do placeholder. None = inferir de model.config.
        grid_shape: (H, W) do grid visual. None = inferir como sqrt(N).
        device: 'cuda' ou 'cpu'.
        return_raw: se True, anexa atencao bruta sobre todas as keys (debug).

    Returns:
        VLMAttentionResult com `attn_grid` em shape (T, H, W).
    """
    notes: list[str] = []

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    if hasattr(model, "dtype") and "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    input_ids: torch.Tensor = inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])

    img_tok_id = _resolve_image_token_id(model, image_token_id)
    placeholder_pos = _find_placeholder_position(input_ids, img_tok_id)
    notes.append(f"placeholder_position={placeholder_pos}, prompt_len={prompt_len}")

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=True,
        output_attentions=True,
    )

    if not getattr(output, "attentions", None):
        raise RuntimeError(
            "model.generate retornou sem `attentions`. Confirme que o modelo "
            "suporta output_attentions=True."
        )

    first_layer_step0 = output.attentions[0][0]
    kv_len_0 = int(first_layer_step0.shape[-1])
    n_visual_tokens = kv_len_0 - (prompt_len - 1)
    notes.append(
        f"kv_len_step0={kv_len_0}, n_visual_tokens={n_visual_tokens} "
        f"(prompt expandido = prompt_len - 1 + N)"
    )

    if n_visual_tokens <= 0:
        raise RuntimeError(
            f"n_visual_tokens nao positivo ({n_visual_tokens}). "
            f"kv_len_0={kv_len_0}, prompt_len={prompt_len}. "
            "Pode nao haver expansao visual nesta arquitetura."
        )

    visual_start = placeholder_pos
    visual_end = placeholder_pos + n_visual_tokens

    if grid_shape is None:
        grid_shape = _estimate_grid_shape(n_visual_tokens)
        notes.append(f"grid_shape inferido={grid_shape}")
    H, W = grid_shape
    expected_count = H * W

    full_seq = output.sequences[0]
    new_tokens = full_seq[prompt_len:]
    tokenizer = getattr(processor, "tokenizer", processor)
    caption = tokenizer.decode(new_tokens, skip_special_tokens=True)
    generated_strs = tokenizer.convert_ids_to_tokens(new_tokens.tolist())

    attn_grids: list[np.ndarray] = []
    raw_per_step: list[np.ndarray] = []

    for t, step_attns in enumerate(output.attentions):
        if step_attns is None:
            continue

        layer_attn = _aggregate_layers(step_attns, layer_aggregation)
        head_attn = _aggregate_heads(layer_attn, head_aggregation)
        # head_attn: (batch=1, q_len, kv_len). q_pos=-1 funciona tanto para
        # prefill (step 0, q_len=full) quanto para incremental (step>0, q_len=1).
        last_query = head_attn[0, -1, :]

        visual_slice = last_query[visual_start:visual_end]
        if visual_slice.shape[0] != expected_count:
            if visual_slice.shape[0] > expected_count:
                if t == 0:
                    notes.append(
                        f"Slice visual ({visual_slice.shape[0]}) maior que H*W ({expected_count}); "
                        "truncando para os ultimos H*W elementos (assumindo CLS no inicio)."
                    )
                visual_slice = visual_slice[-expected_count:]
            else:
                raise RuntimeError(
                    f"step {t}: slice visual tem {visual_slice.shape[0]} elementos, "
                    f"esperado {expected_count} (grid {grid_shape})."
                )

        grid = visual_slice.float().cpu().numpy().reshape(H, W)
        attn_grids.append(grid)
        if return_raw:
            raw_per_step.append(last_query.float().cpu().numpy())

    attn_grid = np.stack(attn_grids, axis=0) if attn_grids else np.empty((0, H, W))

    if len(generated_strs) > attn_grid.shape[0]:
        generated_strs = generated_strs[: attn_grid.shape[0]]

    return VLMAttentionResult(
        caption=caption,
        generated_tokens=generated_strs,
        attn_grid=attn_grid,
        visual_token_range=(visual_start, visual_end),
        grid_shape=grid_shape,
        n_visual_tokens=n_visual_tokens,
        notes=notes,
        raw_attn_per_step=raw_per_step if return_raw else None,
    )

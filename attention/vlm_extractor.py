"""Extracao de mapas de atencao visual em VLMs tipo TinyLLaVA / ViTucano.

Premissa arquitetural
---------------------
ViTucano = encoder visual (SigLIP/CLIP) + projetor MLP + LLM Tucano.
No `input_ids` existe **um** placeholder de imagem (`<image>` -> id especial).
Durante o forward do LLM, esse placeholder e EXPANDIDO em N embeddings
visuais contiguos. O LLM faz self-attention causal sobre a sequencia
expandida, atendendo aos N tokens visuais como atendaria a tokens de texto.

A interface oficial de TinyLLaVA NAO usa um `AutoProcessor` unificado;
sao tokenizer + image_processor separados. O `model.generate` espera
o argumento `images=` (nao `pixel_values=`).

Estrategia de extracao
----------------------
1. Formatar o prompt com `<image>\n{user_prompt}` aplicando o chat_template
   do tokenizer (se houver) ou um template TinyLLaVA padrao como fallback.
2. Tokenizar o texto e processar a imagem separadamente.
3. Localizar a posicao `pos` do placeholder em `input_ids`.
4. Rodar `model.generate(input_ids=..., images=..., image_sizes=...,
   output_attentions=True, return_dict_in_generate=True)`.
5. Inferir N a partir do `kv_len` da primeira camada de atencao do passo 0:
       N = kv_len_0 - (prompt_len - 1)
6. Para cada passo t, agregar camadas e cabecas, pegar a ultima query
   (a que prediz o token gerado) e fatiar [pos, pos+N) nas keys.
7. Reshape para o grid espacial (H, W) com H*W = N.

Limites conhecidos
------------------
- Suporta 1 imagem por inferencia.
- Suporta apenas geracao greedy (num_beams=1).
- Assume que o vision encoder produz tokens em ordem row-major.
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

DEFAULT_IMAGE_TOKEN = "<image>"
FALLBACK_PROMPT_TEMPLATE = "USER: {content}\nASSISTANT:"


@dataclass
class VLMAttentionResult:
    caption: str
    generated_tokens: list[str]
    attn_grid: np.ndarray
    visual_token_range: tuple[int, int]
    grid_shape: tuple[int, int]
    n_visual_tokens: int
    formatted_prompt: str = ""
    notes: list[str] = field(default_factory=list)
    raw_attn_per_step: list[np.ndarray] | None = None


def inspect_vlm_architecture(model, tokenizer) -> dict[str, Any]:
    """Coleta metadados arquiteturais para diagnostico.

    Inspeciona o modelo e o tokenizer (image_processor nao e necessario aqui).
    """
    info: dict[str, Any] = {}
    info["model_class"] = type(model).__name__
    info["torch_dtype"] = str(getattr(model, "dtype", "?"))

    image_token_id = (
        getattr(model.config, "image_token_id", None)
        or getattr(model.config, "image_token_index", None)
    )
    info["image_token_id (config)"] = image_token_id

    if hasattr(tokenizer, "convert_tokens_to_ids"):
        tid = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        info["image_token_id (tokenizer)"] = tid if tid != tokenizer.unk_token_id else None

    info["tokenizer_class"] = type(tokenizer).__name__
    info["chat_template_set"] = bool(getattr(tokenizer, "chat_template", None))

    vision_tower = _resolve_vision_tower(model)
    if vision_tower is not None:
        info["vision_tower_class"] = type(vision_tower).__name__
        v_cfg = getattr(vision_tower, "config", None)
        if v_cfg is not None:
            info["vision_image_size"] = getattr(v_cfg, "image_size", None)
            info["vision_patch_size"] = getattr(v_cfg, "patch_size", None)
            info["vision_hidden_size"] = getattr(v_cfg, "hidden_size", None)

    llm = _resolve_language_model(model)
    if llm is not None and hasattr(llm, "config"):
        info["llm_class"] = type(llm).__name__
        info["llm_n_layers"] = getattr(llm.config, "num_hidden_layers", None)
        info["llm_n_heads"] = getattr(llm.config, "num_attention_heads", None)
        info["llm_hidden_size"] = getattr(llm.config, "hidden_size", None)

    info["has_chat_method"] = hasattr(model, "chat")
    info["forward_accepts_images"] = "images" in str(getattr(model.forward, "__doc__", "") or "") \
        or any("images" in str(p) for p in _get_forward_params(model))

    return info


def _resolve_vision_tower(model):
    candidates = [
        getattr(model, "vision_tower", None),
        getattr(model, "vision_model", None),
        getattr(getattr(model, "model", None), "vision_tower", None),
        getattr(getattr(model, "base_model", None), "vision_tower", None),
    ]
    for c in candidates:
        if c is not None:
            return c
    return None


def _resolve_language_model(model):
    candidates = [
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(getattr(model, "base_model", None), "language_model", None),
    ]
    for c in candidates:
        if c is not None:
            return c
    return None


def _get_forward_params(model):
    import inspect

    try:
        return list(inspect.signature(model.forward).parameters.keys())
    except (TypeError, ValueError):
        return []


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


def _resolve_image_token_id(model, tokenizer, override: int | None) -> int:
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
    if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
        tid = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
            return int(tid)
    raise ValueError(
        "image_token_id nao encontrado em model.config nem no tokenizer "
        f"(token '{DEFAULT_IMAGE_TOKEN}'). Passar explicitamente via argumento."
    )


def _format_prompt(prompt: str, tokenizer, image_token: str = DEFAULT_IMAGE_TOKEN) -> str:
    """Aplica o chat_template do tokenizer com `<image>` prefixado.

    Fallback: template TinyLLaVA basico ("USER: {content}\\nASSISTANT:").
    """
    user_content = f"{image_token}\n{prompt}"
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return FALLBACK_PROMPT_TEMPLATE.format(content=user_content)


def _find_placeholder_position(input_ids: torch.Tensor, image_token_id: int) -> int:
    ids = input_ids[0].tolist()
    positions = [i for i, t in enumerate(ids) if t == image_token_id]
    if len(positions) == 0:
        raise ValueError(
            f"Placeholder de imagem (id={image_token_id}) nao encontrado em input_ids. "
            "O chat_template pode ter removido ou o tokenizer nao reconheceu '<image>' como special."
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
    tokenizer,
    image_processor,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int = 30,
    layer_aggregation: LayerAggregation = "last",
    head_aggregation: HeadAggregation = "mean",
    image_token_id: int | None = None,
    image_token: str = DEFAULT_IMAGE_TOKEN,
    grid_shape: tuple[int, int] | None = None,
    device: str = "cuda",
    return_raw: bool = False,
) -> VLMAttentionResult:
    """Gera caption e extrai mapas de atencao visual por token gerado.

    Args:
        model: VLM tipo TinyLLaVA / ViTucano em eval.
        tokenizer: tokenizer carregado via `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.
        image_processor: processor de imagem via `AutoImageProcessor.from_pretrained(...)`.
        image: PIL.Image em RGB.
        prompt: texto do usuario (sem `<image>` — ele e prefixado automaticamente).
        max_new_tokens: comprimento maximo da geracao.
        layer_aggregation: 'last' | 'first' | 'mean' sobre as camadas do LLM.
        head_aggregation: 'mean' | 'max' sobre as cabecas de atencao.
        image_token_id: id do placeholder. None = inferir.
        image_token: string do placeholder. Default '<image>'.
        grid_shape: (H, W) do grid visual. None = inferir como sqrt(N).
        device: 'cuda' ou 'cpu'.
        return_raw: se True, anexa atencao bruta sobre todas as keys (debug).
    """
    notes: list[str] = []

    img_tok_id = _resolve_image_token_id(model, tokenizer, image_token_id)
    notes.append(f"image_token_id={img_tok_id}")

    formatted = _format_prompt(prompt, tokenizer, image_token=image_token)
    notes.append(f"prompt_formatado={formatted!r}")

    text_inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_ids: torch.Tensor = text_inputs["input_ids"]
    prompt_len = int(input_ids.shape[1])

    image_inputs = image_processor(image, return_tensors="pt")
    pixel_values: torch.Tensor = image_inputs["pixel_values"].to(device)
    if hasattr(model, "dtype"):
        pixel_values = pixel_values.to(model.dtype)

    placeholder_pos = _find_placeholder_position(input_ids, img_tok_id)
    notes.append(f"placeholder_pos={placeholder_pos}, prompt_len={prompt_len}")

    gen_kwargs: dict[str, Any] = dict(
        input_ids=input_ids,
        images=pixel_values,
        image_sizes=[image.size],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=True,
        output_attentions=True,
    )
    if "attention_mask" in text_inputs:
        gen_kwargs["attention_mask"] = text_inputs["attention_mask"]

    output = model.generate(**gen_kwargs)

    if not getattr(output, "attentions", None):
        raise RuntimeError(
            "model.generate retornou sem `attentions`. Confirme suporte a output_attentions=True."
        )

    first_layer_step0 = output.attentions[0][0]
    kv_len_0 = int(first_layer_step0.shape[-1])
    n_visual_tokens = kv_len_0 - (prompt_len - 1)
    notes.append(
        f"kv_len_step0={kv_len_0}, n_visual_tokens={n_visual_tokens}"
    )

    if n_visual_tokens <= 0:
        raise RuntimeError(
            f"n_visual_tokens nao positivo ({n_visual_tokens}). "
            f"kv_len_0={kv_len_0}, prompt_len={prompt_len}."
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
    caption = tokenizer.decode(new_tokens, skip_special_tokens=True)
    generated_strs = tokenizer.convert_ids_to_tokens(new_tokens.tolist())

    attn_grids: list[np.ndarray] = []
    raw_per_step: list[np.ndarray] = []

    for t, step_attns in enumerate(output.attentions):
        if step_attns is None:
            continue

        layer_attn = _aggregate_layers(step_attns, layer_aggregation)
        head_attn = _aggregate_heads(layer_attn, head_aggregation)
        last_query = head_attn[0, -1, :]

        visual_slice = last_query[visual_start:visual_end]
        if visual_slice.shape[0] != expected_count:
            if visual_slice.shape[0] > expected_count:
                if t == 0:
                    notes.append(
                        f"slice ({visual_slice.shape[0]}) > H*W ({expected_count}); "
                        "truncando para os ultimos H*W (assumindo CLS)."
                    )
                visual_slice = visual_slice[-expected_count:]
            else:
                raise RuntimeError(
                    f"step {t}: slice visual {visual_slice.shape[0]} elementos, "
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
        formatted_prompt=formatted,
        notes=notes,
        raw_attn_per_step=raw_per_step if return_raw else None,
    )

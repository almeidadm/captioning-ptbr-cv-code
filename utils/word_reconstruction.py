"""Reconstrucao de palavras a partir de tokens BPE/SentencePiece.

Modulo de apoio ao protocolo de inspecao qualitativa de heatmaps de
atencao (`notas/metodologia/protocolo-inspecao-heatmap.md`).

Trata trees centrais:
1. Subwords (e.g. `_chap` + `eu` -> "chapeu") sao agregados na unidade
   de classificacao do protocolo: a palavra reconstruida.
2. Stopwords sao identificadas via lista PT-BR fixa; categorias
   permitidas para elas no protocolo sao restritas a DIFUSO/ARTEFATO.
3. Variabilidade de atencao entre subwords ("drift") e diagnostico
   de instabilidade autoregressiva intra-palavra.

LlamaTokenizer/SentencePiece marca inicio de palavra com U+2581
(`_`, "lower one eighth block"). Subwords seguintes nao tem prefixo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


SP_WORD_PREFIX = "▁"  # SentencePiece: inicio de palavra


# Stopwords PT-BR (lista operacional do protocolo).
# Combina NLTK PT + adicoes especificas para captioning.
STOPWORDS_PT = frozenset({
    # Artigos
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    # Preposicoes
    "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "com", "para", "por", "sobre", "sob",
    "entre", "ate", "desde", "ante", "apos", "perante",
    # Conjuncoes
    "e", "ou", "mas", "que", "porque", "se", "embora", "como",
    "pois", "porem", "todavia", "contudo", "entao",
    # Pronomes (subset)
    "ele", "ela", "eles", "elas", "isto", "isso", "aquilo",
    "este", "esta", "esse", "essa", "aquele", "aquela",
    # Verbos copula / auxiliares (subset)
    "e", "eh", "era", "eram", "sao", "estao", "esta", "estava",
    "ser", "estar", "ter", "tem", "tinha",
    # Adverbios neutros
    "nao", "sim", "ja", "ainda", "muito", "pouco", "tambem",
    "apenas", "so", "tao", "bem", "mal",
})


# Tokens especiais: classificar como N/A no protocolo
SPECIAL_TOKENS = frozenset({
    "<eos>", "</s>", "<bos>", "<s>", "<unk>", "<pad>",
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
})


@dataclass
class ReconstructedWord:
    """Palavra reconstruida a partir de subwords contiguos."""

    text: str
    subword_indices: list[int] = field(default_factory=list)
    subword_tokens: list[str] = field(default_factory=list)

    @property
    def n_subwords(self) -> int:
        return len(self.subword_indices)

    @property
    def is_multi_subword(self) -> bool:
        return self.n_subwords > 1


def _strip_sp_prefix(token: str) -> str:
    """Remove prefixo SentencePiece se presente."""
    if token.startswith(SP_WORD_PREFIX):
        return token[len(SP_WORD_PREFIX):]
    return token


def reconstruct_words(tokens: Iterable[str]) -> list[ReconstructedWord]:
    """Agrupa subwords contiguos em palavras.

    Uma palavra inicia em todo token com prefixo `_` (U+2581) ou no
    primeiro token da sequencia. Tokens subsequentes sem prefixo
    sao anexados a palavra atual.

    Tokens especiais (vide `SPECIAL_TOKENS`) sao tratados como
    palavras isoladas independente de prefixo.

    Args:
        tokens: lista de tokens BPE/SP do tokenizer (string).

    Returns:
        Lista de `ReconstructedWord` na ordem de geracao.
    """
    words: list[ReconstructedWord] = []
    current: ReconstructedWord | None = None

    for idx, tok in enumerate(tokens):
        is_special = tok in SPECIAL_TOKENS
        starts_word = tok.startswith(SP_WORD_PREFIX) or is_special or current is None

        if starts_word:
            if current is not None:
                words.append(current)
            current = ReconstructedWord(
                text=_strip_sp_prefix(tok) if not is_special else tok,
                subword_indices=[idx],
                subword_tokens=[tok],
            )
        else:
            assert current is not None
            current.text += _strip_sp_prefix(tok)
            current.subword_indices.append(idx)
            current.subword_tokens.append(tok)

    if current is not None:
        words.append(current)

    return words


def aggregate_attention(
    attn_grid: np.ndarray,
    word: ReconstructedWord,
    strategy: str = "mean",
) -> np.ndarray:
    """Agrega atencao dos subwords de uma palavra.

    Args:
        attn_grid: array (T, H, W) com atencao por subword.
        word: palavra reconstruida com `subword_indices`.
        strategy: "mean" (default), "max", "first", "sum".

    Returns:
        Array (H, W) com atencao agregada da palavra.
    """
    indices = word.subword_indices
    slices = attn_grid[indices]  # (k, H, W)

    if strategy == "mean":
        return slices.mean(axis=0)
    if strategy == "max":
        return slices.max(axis=0)
    if strategy == "first":
        return slices[0]
    if strategy == "sum":
        return slices.sum(axis=0)
    raise ValueError(f"strategy desconhecida: {strategy!r}")


def is_stopword(word_text: str) -> bool:
    """Verifica se a palavra (ja sem prefixo SP) e stopword PT-BR."""
    return word_text.lower().strip() in STOPWORDS_PT


def is_special(word_text: str) -> bool:
    """Verifica se e token especial (eos, unk, etc)."""
    return word_text in SPECIAL_TOKENS


def classify_word_type(
    word_text: str,
    consensual_nouns: set[str] | None = None,
    isolated_nouns: set[str] | None = None,
) -> str:
    """Classificacao automatica preliminar do tipo da palavra.

    Categorias retornadas (alinhadas com o protocolo):
    - "especial": eos, unk, etc
    - "stopword": palavra funcional PT-BR
    - "substantivo-cons": substantivo presente em >= 2 referencias
    - "substantivo-iso": substantivo presente em 1 referencia
    - "substantivo-nao-coberto": substantivo nao mencionado em nenhuma
      referencia (candidato a HALLUCINADO)
    - "outro": adjetivos, verbos, palavras que nao se encaixam acima

    Heuristica simples — usuario revisa manualmente. POS tagging e
    deixado fora para nao introduzir dependencia (spacy, nltk).

    Args:
        word_text: palavra reconstruida (sem prefixo SP).
        consensual_nouns: substantivos com >= 2 ocorrencias nas
            captions de referencia (lowercase, sem acentos).
        isolated_nouns: substantivos com 1 ocorrencia.

    Returns:
        String com a categoria sugerida.
    """
    text = word_text.lower().strip()

    if is_special(word_text):
        return "especial"
    if is_stopword(word_text):
        return "stopword"
    if consensual_nouns and text in consensual_nouns:
        return "substantivo-cons"
    if isolated_nouns and text in isolated_nouns:
        return "substantivo-iso"
    # Heuristica fraca: se tem >= 4 caracteres e nao e stopword, pode
    # ser substantivo. Marcar como "nao-coberto" se nao esta nas
    # referencias. Usuario deve confirmar/ajustar.
    if len(text) >= 4 and consensual_nouns is not None:
        return "substantivo-nao-coberto"
    return "outro"


def compute_drift_flag(
    attn_grid: np.ndarray,
    word: ReconstructedWord,
    grid_shape: tuple[int, int] | None = None,
) -> bool:
    """Diagnostica drift autoregressivo intra-palavra.

    Drift_alto = sim quando os subwords da palavra tem picos em
    quadrantes diferentes da grade 3x3 da imagem.

    Para palavras com 1 subword: retorna False (nao aplicavel).

    Args:
        attn_grid: array (T, H, W).
        word: palavra reconstruida.
        grid_shape: (H, W) ou None (infere de attn_grid.shape[1:]).

    Returns:
        True se ha drift entre subwords; False caso contrario.
    """
    if not word.is_multi_subword:
        return False

    H, W = grid_shape if grid_shape is not None else attn_grid.shape[1:]

    quadrants: list[tuple[int, int]] = []
    for idx in word.subword_indices:
        slice_ = attn_grid[idx]  # (H, W)
        flat_argmax = int(slice_.argmax())
        peak_h, peak_w = flat_argmax // W, flat_argmax % W
        # Mapeia para grade 3x3
        q_h = min(2, int(peak_h * 3 / H))
        q_w = min(2, int(peak_w * 3 / W))
        quadrants.append((q_h, q_w))

    return len(set(quadrants)) > 1


def extract_nouns_from_captions(captions: Iterable[str]) -> dict[str, int]:
    """Extrai contagem aproximada de substantivos das captions.

    Heuristica simples: conta palavras de >= 4 caracteres que nao
    sao stopwords. Retorna dict {palavra_lower: n_captions_em_que_aparece}.

    NAO substitui POS tagging — e uma aproximacao operacional.
    Usuario deve revisar e remover falsos positivos (verbos, adjetivos).

    Args:
        captions: iteravel de strings.

    Returns:
        Dict com contagem por palavra.
    """
    import re
    counts: dict[str, int] = {}
    for caption in captions:
        # tokenizacao simples; mantem so letras (sem digitos/pontuacao)
        words_in_caption = set()
        for raw in re.findall(r"[A-Za-zÀ-ÿ]+", caption.lower()):
            if len(raw) >= 4 and not is_stopword(raw):
                words_in_caption.add(raw)
        for w in words_in_caption:
            counts[w] = counts.get(w, 0) + 1
    return counts


def build_word_table(
    tokens: list[str],
    attn_grid: np.ndarray,
    captions: Iterable[str] | None = None,
    grid_shape: tuple[int, int] | None = None,
) -> list[dict]:
    """Pipeline completo: tokens -> tabela pronta para a Etapa 2.

    Cada linha do retorno tem:
        w: indice da palavra
        text: palavra reconstruida
        subwords: lista de subwords originais
        subword_indices: lista de indices em attn_grid
        n_refs: ocorrencias da palavra nas captions (0 se nao
            cobertas; -1 se captions nao fornecidas)
        tipo_sugerido: categoria automatica preliminar
        drift_alto: bool (False se 1 subword)

    Args:
        tokens: lista de tokens (T elementos).
        attn_grid: array (T, H, W).
        captions: iteravel de captions de referencia (5+5 do dataset).
            Se None, n_refs = -1 e tipo nao distingue cons/iso.
        grid_shape: opcional (H, W).

    Returns:
        Lista de dicts ordenada por w.
    """
    words = reconstruct_words(tokens)

    if captions is not None:
        noun_counts = extract_nouns_from_captions(captions)
        consensual = {w for w, c in noun_counts.items() if c >= 2}
        isolated = {w for w, c in noun_counts.items() if c == 1}
    else:
        noun_counts = {}
        consensual = None  # type: ignore[assignment]
        isolated = None  # type: ignore[assignment]

    rows: list[dict] = []
    for w_idx, word in enumerate(words):
        text_lower = word.text.lower().strip()
        n_refs = noun_counts.get(text_lower, 0) if captions is not None else -1
        rows.append({
            "w": w_idx,
            "text": word.text,
            "subwords": word.subword_tokens,
            "subword_indices": word.subword_indices,
            "n_refs": n_refs,
            "tipo_sugerido": classify_word_type(
                word.text,
                consensual_nouns=consensual,
                isolated_nouns=isolated,
            ),
            "drift_alto": compute_drift_flag(attn_grid, word, grid_shape),
        })
    return rows


def render_table_markdown(rows: list[dict]) -> str:
    """Renderiza a tabela como markdown para colar no .md."""
    lines = [
        "| w | Palavra | Subwords (idx) | Tipo | n_refs | drift_alto |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        idx_str = ",".join(str(i) for i in row["subword_indices"])
        n_refs = row["n_refs"]
        n_refs_str = "-" if n_refs == -1 else str(n_refs)
        drift = "sim" if row["drift_alto"] else ("-" if len(row["subword_indices"]) == 1 else "nao")
        lines.append(
            f"| {row['w']} | {row['text']} | {idx_str} | "
            f"{row['tipo_sugerido']} | {n_refs_str} | {drift} |"
        )
    return "\n".join(lines)

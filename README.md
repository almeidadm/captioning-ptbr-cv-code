# captioning-ptbr-cv-code

Pacote Python para extracao e analise quantitativa de mapas de atencao visual em modelos de image captioning para portugues brasileiro (VED + VLM). Implementacao da disciplina de Visao Computacional do PPGCC/UNIFESP, derivada de Bromonschenkel et al. (2026).

Repositorio de **codigo apenas**. Notas, decisoes metodologicas e relatorios vivem no vault Obsidian que consome este pacote.

## Instalacao

Em Google Colab:

```python
!git clone https://github.com/<usuario>/captioning-ptbr-cv-code.git /content/codigo
import sys
sys.path.insert(0, "/content")
import codigo
```

Localmente, clonar dentro de qualquer diretorio cujo pai esteja em `sys.path`:

```bash
git clone https://github.com/<usuario>/captioning-ptbr-cv-code.git codigo
python -c "import codigo; print(codigo.__file__)"
```

> Importante: o diretorio clonado **precisa se chamar `codigo`** para o `import codigo` funcionar. Use `git clone <url> codigo` (segundo argumento explicito) ou renomeie apos clonar.

## API resumida

```python
from codigo.io import load_sample_from_hf
from codigo.attention.vlm_extractor import extract_vlm_attention, inspect_vlm_architecture
from codigo.viz import overlay_heatmap, plot_attention_grid
from codigo.utils.paths import get_drive_paths
```

## Estrutura

| Modulo | Conteudo |
|---|---|
| `codigo.attention` | Extratores de atencao para VED (cross-attention) e VLM (atencao do LLM sobre tokens visuais) |
| `codigo.io` | Carregamento de amostras dos datasets Flickr30K-PT-BR (traduzido e nativo) |
| `codigo.viz` | Overlay de heatmap em PIL e grid de subplots por token |
| `codigo.utils` | Paths persistentes em Google Drive |

## Modelos suportados (PoC S1)

- VED: `laicsiifes/swin-distilbertimbau`, `laicsiifes/swin-gportuguese-2`
- VLM: `laicsiifes/vitucano-1b-flickr30k_pt`, `laicsiifes/vitucano-1b-flickr30k_pt_human_generated`

## Datasets

- `laicsiifes/flickr30k-pt-br` (traducao automatica)
- `laicsiifes/flickr30k-pt-br-human-generated` (captions nativas)

## Dependencias

Pinadas no notebook S1 do projeto consumidor:

```
transformers==4.45.2
accelerate==1.0.1
safetensors==0.4.5
datasets==3.0.1
pillow>=10.0
matplotlib>=3.8
numpy>=1.26
pyarrow>=15.0
sentencepiece>=0.2
protobuf>=4.25
```

## Licenca

A definir.

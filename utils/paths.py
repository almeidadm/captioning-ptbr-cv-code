"""Paths persistentes em Google Drive (Colab) ou local.

Convencao alinhada ao escopo refinado v3, secao "Estrutura de persistencia".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DRIVE_ROOT = Path("/content/drive/MyDrive/captioning-ptbr-cv")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    checkpoints: Path
    dados: Path
    flickr_traduzido: Path
    flickr_nativo: Path
    deteccoes: Path
    atencao: Path
    resultados: Path
    metricas: Path
    predicoes: Path
    anotacao_manual: Path

    def make_dirs(self) -> None:
        """Cria toda a arvore de pastas se ainda nao existir.

        Paths com sufixo de arquivo (ex: .parquet, .json) tem o pai
        criado em vez do proprio caminho.
        """
        for value in self.__dict__.values():
            if not isinstance(value, Path):
                continue
            target = value.parent if value.suffix else value
            target.mkdir(parents=True, exist_ok=True)


def get_drive_paths(root: Path | str | None = None) -> ProjectPaths:
    """Retorna conjunto de paths derivados de uma raiz.

    Em Colab, usar root=None (default = DRIVE_ROOT).
    Em ambiente local, passar uma raiz alternativa para testes.
    """
    base = Path(root) if root is not None else DRIVE_ROOT
    return ProjectPaths(
        root=base,
        checkpoints=base / "checkpoints",
        dados=base / "dados",
        flickr_traduzido=base / "dados" / "flickr30k-pt-br",
        flickr_nativo=base / "dados" / "flickr30k-native",
        deteccoes=base / "dados" / "deteccoes",
        atencao=base / "atencao",
        resultados=base / "resultados",
        metricas=base / "resultados" / "metricas_atencao.parquet",
        predicoes=base / "resultados" / "predicoes.json",
        anotacao_manual=base / "resultados" / "anotacao_manual",
    )

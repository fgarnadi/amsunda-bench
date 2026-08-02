import random
import re
from collections.abc import Callable, Generator
from functools import partialmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
from pyterrier_dr import SBertBiEncoder
from sentence_transformers import CrossEncoder


# monkeypatch
class NormalizedSBertBiEncoder(SBertBiEncoder):
    encode_docs = partialmethod(
        SBertBiEncoder.encode_docs,
        normalize_embeddings=True,
    )

    encode_queries = partialmethod(
        SBertBiEncoder.encode_queries,
        normalize_embeddings=True,
    )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # https://docs.pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
    torch.set_float32_matmul_precision("high")


def write_ndjson(df: pl.DataFrame, path: Path, cols: list[str] = []):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.select(cols or df.columns).write_ndjson(path)


def write_csv(
    df: pl.DataFrame,
    path: Path,
    cols: list[str] = [],
    sep: str = ",",
    include_header=True,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.select(cols or df.columns).write_csv(
        path, separator=sep, include_header=include_header
    )


def make_ngrams(tokens: list[str], n: int, sep: str = "_") -> list[str]:
    return [sep.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def white_space_tokenizer(text: str) -> list[str]:
    return re.findall(r"\w+", text)


def tokenize(
    text: str | None,
    n_max: int = 1,
    tokenizer: Callable[[str], list[str]] = white_space_tokenizer,
) -> list[str]:
    if text is None:
        return []

    toks = tokenizer(text.lower())
    out = toks[:]  # unigrams
    for n in range(2, n_max + 1):
        out += make_ngrams(toks, n)
    return out


def iter_from_csv(
    path: Path,
    transformer: dict[str, Callable[[Any], Any]] | None = None,
    rename: dict[str, str] | None = None,
) -> Generator[dict[str, Any]]:
    transformer = transformer or {}
    rename = rename or {}

    df = pl.read_csv(path, encoding="utf8")

    for row in df.iter_rows(named=True):
        for key, transform in transformer.items():
            row[key] = transform(row)

        for old, new in rename.items():
            row[new] = row.pop(old)

        yield row


def crossencoder_apply(model: str, verbose: bool = False):
    ce = CrossEncoder(model, num_labels=1)

    def apply(inp: pd.DataFrame):
        return ce.predict(
            list(zip(inp["query"].values, inp["text"].values)),
            show_progress_bar=verbose,
        )

    return apply

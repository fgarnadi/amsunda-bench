import argparse
import random
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any

import pyterrier as pt
from utils import iter_from_csv, tokenize


def preprocess(
    row: dict[str, Any],
    n_max: int = 1,
    join_title: bool = False,
) -> str:
    tokens = tokenize(row["text"], n_max)

    if join_title:
        tokens = tokenize(row["title"]) + tokens

    return " ".join(tokens)


def pretokenize(
    row: dict[str, Any],
    n_max: int = 1,
    join_title: bool = False,
) -> dict[str, int]:
    tokens = tokenize(row["text"], n_max)

    if join_title:
        tokens = tokenize(row["title"]) + tokens

    # {'docno' : 'd1', 'toks' : {'a' : 1, 'aa' : 2}}
    return Counter(tokens)


def main(args: Namespace):
    pt.java.init()

    input_dir = Path(args.input_dir)
    corpus_path = input_dir / "corpus.csv"

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Override the pre-processing to use whitespace only
    indexer = pt.IterDictIndexer(
        index_dir.absolute().as_posix(),
        pretokenised=True,  # handled manually,
        stemmer=None,
        stopwords=None,
        # tokeniser=pt.TerrierTokeniser.whitespace,
        meta={
            "docno": 50,
            "text": 8192,
        },
        properties={
            "max.term.length": "128",
        },
        overwrite=True,
    )

    # index
    docs = iter_from_csv(
        corpus_path,
        transformer={
            "text": lambda row: preprocess(row, args.ngram, args.join_title),
            "toks": lambda row: Counter(row["text"].split()),
        },
    )
    indexref = indexer.index(docs)  # type: ignore

    print(f"Indexing complete. Index reference: {indexref}")
    print()
    print("Sample terms: ")
    index = pt.IndexFactory.of(indexref)  # type: ignore
    lexicon = [entry.getKey() for entry in index.getLexicon()]
    for term in random.sample(lexicon, 10):
        print(f"\t{term}")

    print()
    print(index.getCollectionStatistics())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="dataset/baseline")
    ap.add_argument("--index_dir", default="indexes/terrier/bm25")
    ap.add_argument("--ngram", type=int, default=1)
    ap.add_argument("--join_title", action="store_true", default=False)
    args = ap.parse_args()

    main(args)

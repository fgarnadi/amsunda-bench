import argparse
from argparse import Namespace
from pathlib import Path
from typing import Any

import pyterrier as pt
import pyterrier_dr as ptdr
from utils import NormalizedSBertBiEncoder, iter_from_csv, seed_everything


def preprocess(
    row: dict[str, Any],
    join_title: bool = False,
) -> str:
    text = row["text"]
    if join_title:
        text = row["title"] + text

    return text


def main(args: Namespace):
    pt.java.init()

    seed_everything(args.seed)

    input_dir = Path(args.input_dir)
    corpus_path = input_dir / "corpus.csv"

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    model = NormalizedSBertBiEncoder(
        args.model_path,
        batch_size=16,
        verbose=args.verbose,
    )

    index = ptdr.FlexIndex(
        index_dir.absolute().as_posix(),
        sim_fn=ptdr.SimFn.cos,
    )

    # index
    docs = iter_from_csv(
        corpus_path,
        transformer={
            "text": lambda row: preprocess(row, args.join_title),
        },
    )
    pipeline = (model >> index.indexer(mode="overwrite")).compile()  # type: ignore

    indexref = pipeline.index(docs)  # type: ignore

    print(f"Indexing complete. Index reference: {indexref}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path", default="models/mmbert-amsunda-biencoder/final"
    )
    ap.add_argument("--input_dir", default="dataset/baseline")
    ap.add_argument("--index_dir", default="indexes/terrier/bi-encoder")
    ap.add_argument("--join_title", action="store_true", default=False)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", default=42)
    args = ap.parse_args()

    main(args)

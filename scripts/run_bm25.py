import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd
import polars as pl
import pyterrier as pt
from constants import TOPK
from utils import tokenize


def main(args: Namespace):
    pt.java.init()

    output_path = Path(args.output_path)

    input_dir = Path(args.input_dir)
    queries_path = input_dir / "queries.csv"

    topics = (
        pl.read_csv(queries_path, encoding="utf8")
        .with_columns(
            pl.col("query")
            .map_elements(
                lambda s: tokenize(s, args.ngram), return_dtype=pl.List(pl.Utf8)
            )
            .list.join(" "),
        )
        .select(["qid", "query"])
        .to_pandas()
    )

    print(f"Index: {args.index_dir}")
    index_path = Path(args.index_dir) / "data.properties"
    index = pt.IndexFactory.of(index_path.absolute().as_posix())  # type: ignore

    print("Init retriever: BM25")
    retriever = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=TOPK,
        controls={
            "bm25.k_1": args.k1,
            "bm25.b": args.b,
        },
        verbose=args.verbose,
    )  # type: ignore

    res: pd.DataFrame = retriever.transform(topics)

    pt.io.write_results(res, output_path.as_posix())

    print(f"Run complete. Ouput: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="dataset/baseline")
    ap.add_argument("--index_dir", default="indexes/terrier/amsunda.bm25")
    ap.add_argument("--output_path", default="runs/bm25.trec")
    ap.add_argument("--ngram", type=int, default=1)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    main(args)

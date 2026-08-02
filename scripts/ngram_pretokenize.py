import argparse
import shutil
from argparse import Namespace
from pathlib import Path

import polars as pl
from utils import tokenize, write_csv


def main(args: Namespace):
    """ "Pretokenize corpus/query into uni+N-grams for BM25."""

    input_dir = Path(args.input_dir)
    corpus_path = input_dir / "corpus.csv"
    query_path = input_dir / "queries.csv"
    qrels_path = input_dir / "test.qrels.csv"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cdf = pl.read_csv(corpus_path, encoding="utf8")
    if args.join_title:
        # merge title + text
        cdf = cdf.with_columns(
            pl.concat_str(
                [
                    pl.col("title").fill_null(""),
                    pl.lit(" "),
                    pl.col("text").fill_null(""),
                ]
            ).alias("text")
        )

    cdf = cdf.with_columns(
        pl.col("text")
        .fill_null("")
        .map_elements(
            lambda s: tokenize(s, args.ngram),
            return_dtype=pl.Utf8,
        )
    ).select(["docno", "text"])

    write_csv(cdf, out_dir / corpus_path.name)  # {docno, text}

    qdf = pl.read_csv(query_path, encoding="utf8")
    qdf = qdf.with_columns(
        pl.col("query")
        .fill_null("")
        .map_elements(lambda s: tokenize(s, args.ngram), return_dtype=pl.Utf8)
    ).select(["qid", "query"])

    write_csv(qdf, out_dir / query_path.name)  # {qid, query}

    # Copy the qrels
    shutil.copy(qrels_path, out_dir / qrels_path.name)

    print("Wrote pretokenized:")
    for file in out_dir.glob("*"):
        print(f"\t{file.as_posix()}")
    if args.join_title:
        print("\t(joined title + text)")
    print(f"\tn-gram max: {args.ngram}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="dataset/baseline")
    ap.add_argument("--out_dir", default="dataset/baseline_ng2")
    ap.add_argument("--ngram", type=int, default=2)
    ap.add_argument("--join_title", action="store_true", default=False)
    args = ap.parse_args()

    main(args)

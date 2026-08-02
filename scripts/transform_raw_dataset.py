import argparse
from argparse import Namespace
from itertools import chain
from pathlib import Path

import polars as pl
from utils import write_csv


def main(args: Namespace):
    """Formatted for PyTerrier Indexer"""

    raw_path = Path(args.input_dir)

    # _id,title,text,title_english,text_english
    corpus_path = raw_path / "corpus.csv"

    # query-id,corpus-id,score
    qrels_path = raw_path / "qrels.csv"

    # _id,text,text_english
    query_path = raw_path / "queries.csv"

    # query,positive,negative,query_english,positive_english,negative_english
    triplet_path = raw_path / "triplet.csv"

    # Read data; no english
    corpus = (
        pl.read_csv(corpus_path, encoding="utf8")
        .drop_nulls()
        .select(
            [
                pl.col("_id").alias("docno"),
                "title",
                "text",
            ]
        )
    )
    triplet = (
        pl.read_csv(triplet_path, encoding="utf8")
        .drop_nulls()
        .select(["query", "positive", "negative"])
    )

    # TREC Format
    queries = (
        pl.read_csv(query_path, encoding="utf8")
        .drop_nulls()
        .select(
            [
                pl.col("_id").alias("qid"),
                pl.col("text").alias("query"),
            ]
        )
    )

    qrels = (
        pl.read_csv(qrels_path, encoding="utf8")
        .drop_nulls()
        .with_columns(pl.lit("Q0").alias("iter"))
        .select(
            [
                pl.col("query-id").alias("qid"),
                "iter",
                pl.col("corpus-id").alias("docno"),
                pl.col("score").alias("label"),
            ]
        )
    )

    print(
        f"Corpus: {corpus.height} docs"
        f" | Triplets: {triplet.height}"
        f" | Queries: {queries.height}"
        f" | Qrels: {qrels.height}"
    )  # fmt: skip

    # Training prep
    train_path = Path(args.out_train_dir)
    write_csv(corpus, train_path / "corpus.csv")
    write_csv(triplet, train_path / "train.triples.csv")

    # Eval prep (unigram)
    base_path = Path(args.out_baseline_dir)
    write_csv(corpus, base_path / "corpus.csv")
    write_csv(queries, base_path / "queries.csv")
    write_csv(
        qrels,
        base_path / "test.qrels.csv",
        sep="\t",
        include_header=False,
    )

    print("Wrote: ")
    for file in chain(train_path.glob("*"), base_path.glob("*")):
        print(f"\t{file.as_posix()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="dataset/raw")
    ap.add_argument("--out_train_dir", default="dataset/training")
    ap.add_argument("--out_baseline_dir", default="dataset/baseline")
    args = ap.parse_args()

    main(args)

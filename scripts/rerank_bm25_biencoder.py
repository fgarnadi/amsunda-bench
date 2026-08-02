import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd
import polars as pl
import pyterrier as pt
import pyterrier_dr as ptdr
from constants import TOPK
from utils import seed_everything, tokenize


def main(args: Namespace):
    pt.java.init()

    seed_everything(args.seed)

    output_path = Path(args.output_path)

    input_dir = Path(args.input_dir)
    queries_path = input_dir / "queries.csv"

    topics = (
        pl.read_csv(queries_path, encoding="utf8")
        .with_columns(
            pl.col("query").alias("query_raw"),
            pl.col("query")
            .map_elements(
                lambda s: tokenize(s, args.ngram), return_dtype=pl.List(pl.Utf8)
            )
            .list.join(" "),
        )
        .select(["qid", "query", "query_raw"])
        .to_pandas()
    )

    print(f"Index: {args.index_dir}")
    index_path = Path(args.index_dir) / "data.properties"
    index = pt.IndexFactory.of(index_path.absolute().as_posix())  # type: ignore

    print(f"Bi-Encoder Index: {args.reranker_index_dir}")
    rerank_index_dir = Path(args.reranker_index_dir)
    rerank_index = ptdr.FlexIndex(
        rerank_index_dir.absolute().as_posix(),
        sim_fn=ptdr.SimFn.cos,
    )

    print("Init retriever: BM25")
    bm25 = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        num_results=TOPK,
        controls={
            "bm25.k_1": args.k1,
            "bm25.b": args.b,
        },
        verbose=args.verbose,
    )  # type: ignore

    print(f"Init reranker: {args.reranker_model_path}")
    be_rerank = ptdr.SBertBiEncoder(
        args.reranker_model_path,
        verbose=args.verbose,
    )

    swap_query = pt.apply.query(
        lambda row: row["query_raw"], verbose=args.verbose
    )

    # 1. Retrieve using bm25
    # 2. Populate the 'query' with the raw value
    # 3. Rerank using Bi-Encoder Index
    pipeline = (
        bm25
        >> swap_query
        >> be_rerank.query_encoder()
        >> rerank_index.np_scorer()  # type: ignore
    ).compile()

    res: pd.DataFrame = pipeline.transform(topics)
    res = res.sort_values(by=["qid", "score"], ascending=[True, False])

    pt.io.write_results(res, output_path.as_posix())

    print(f"Run complete. Ouput: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reranker_model_path", default="models/mmbert-amsunda-biencoder/final"
    )
    ap.add_argument("--input_dir", default="dataset/baseline")
    ap.add_argument("--index_dir", default="indexes/terrier/amsunda.bm25")
    ap.add_argument(
        "--reranker_index_dir", default="indexes/terrier/amsunda.biencoder.flex"
    )
    ap.add_argument("--output_path", default="runs/bm25.trec")
    ap.add_argument("--ngram", type=int, default=1)
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", default=42)
    args = ap.parse_args()

    main(args)

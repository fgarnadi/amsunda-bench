import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd
import polars as pl
import pyterrier as pt
import pyterrier_dr as ptdr
from constants import TOPK
from utils import crossencoder_apply, seed_everything


def main(args: Namespace):
    pt.java.init()

    seed_everything(args.seed)

    output_path = Path(args.output_path)

    input_dir = Path(args.input_dir)
    queries_path = input_dir / "queries.csv"
    corpus_path = input_dir / "corpus.csv"

    topics = (
        pl.read_csv(queries_path, encoding="utf8")
        .select(["qid", "query"])
        .to_pandas()
    )
    corpus = (
        pl.read_csv(corpus_path, encoding="utf8")
        .select(["docno", "text"])
        .to_pandas()
        .set_index("docno")
    )

    print(f"Index: {args.index_dir}")
    index_dir = Path(args.index_dir)
    index = ptdr.FlexIndex(
        index_dir.absolute().as_posix(),
        sim_fn=ptdr.SimFn.cos,
    )

    print(f"Init retriever: {args.model_path}")
    be_retr = ptdr.SBertBiEncoder(
        args.model_path,
        verbose=args.verbose,
    )

    print(f"Init reranker: {args.reranker_model_path}")
    ce_apply = crossencoder_apply(args.model_path)
    ce_rerank = pt.apply.doc_score(
        ce_apply,
        batch_size=32,
        verbose=args.verbose,
    )

    get_text = pt.apply.generic(
        lambda df: df.merge(corpus, on="docno", how="left"),
        verbose=args.verbose,
    )

    pipeline = (
        be_retr
        >> index.np_retriever(num_results=TOPK)  # type: ignore
        >> get_text
        >> ce_rerank
    ).compile()

    res: pd.DataFrame = pipeline.transform(topics)
    res = res.sort_values(by=["qid", "score"], ascending=[True, False])

    pt.io.write_results(res, output_path.as_posix())

    print(f"Run complete. Ouput: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path", default="models/mmbert-amsunda-biencoder/final"
    )
    ap.add_argument(
        "--reranker_model_path",
        default="models/mmbert-amsunda-crossencoder/final",
    )
    ap.add_argument("--input_dir", default="dataset/baseline")
    ap.add_argument(
        "--index_dir", default="indexes/terrier/amsunda.biencoder.flex"
    )
    ap.add_argument("--output_path", default="runs/biencoder.v2.trec")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", default=42)
    args = ap.parse_args()

    main(args)

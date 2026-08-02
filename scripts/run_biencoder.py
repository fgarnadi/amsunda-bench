import argparse
from argparse import Namespace
from pathlib import Path

import polars as pl
import pyterrier as pt
import pyterrier_dr as ptdr
from constants import TOPK
from utils import NormalizedSBertBiEncoder, seed_everything


def main(args: Namespace):
    pt.java.init()

    seed_everything(args.seed)

    output_path = Path(args.output_path)

    input_dir = Path(args.input_dir)
    queries_path = input_dir / "queries.csv"

    topics = (
        pl.read_csv(queries_path, encoding="utf8")
        .select(["qid", "query"])
        .to_pandas()
    )

    print(f"Index: {args.index_dir}")
    index_dir = Path(args.index_dir)
    index = ptdr.FlexIndex(
        index_dir.absolute().as_posix(),
        sim_fn=ptdr.SimFn.cos,
    )

    print(f"Init retriever: {args.model_path}")
    model = NormalizedSBertBiEncoder(
        args.model_path,
        verbose=args.verbose,
    )

    retriever = (model >> index.np_retriever(num_results=TOPK)).compile()  # type: ignore

    res = retriever.transform(topics)

    pt.io.write_results(res, output_path.as_posix())

    print(f"Run complete. Ouput: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path", default="models/mmbert-amsunda-biencoder/final"
    )
    ap.add_argument("--input_dir", default="dataset/baseline")
    ap.add_argument(
        "--index_dir", default="indexes/terrier/amsunda.biencoder.flex"
    )
    ap.add_argument("--output_path", default="runs/biencoder.trec")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", default=42)
    args = ap.parse_args()

    main(args)

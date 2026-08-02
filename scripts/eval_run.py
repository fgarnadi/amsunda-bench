import argparse
from argparse import Namespace
from pathlib import Path

import pandas as pd
import polars as pl
import pyterrier as pt
from ir_measures import MRR, Recall, nDCG
from tqdm import tqdm


def load_run(path):
    # read TREC run into DataFrame with qid, docno, score, rank, system
    df = pt.io.read_results(str(path))
    # keep what ir_measures needs
    return df[["qid", "docno", "score", "rank"]]


def eval_one(run: pd.DataFrame, qrels: pd.DataFrame) -> dict:
    metrics = [
        nDCG @ 1,
        nDCG @ 5,
        nDCG @ 10,
        Recall @ 1,
        Recall @ 5,
        Recall @ 10,
        MRR @ 10,
    ]

    return pt.Evaluate(run, qrels, metrics)


def main(args: Namespace):
    pt.java.init()

    input_dir = Path(args.input_dir)

    qrels_path = input_dir / "test.qrels.csv"
    qrels = pt.io.read_qrels(qrels_path.absolute().as_posix())

    # Directory glob
    if args.run_dir and args.output_dir:
        run_dir = Path(args.run_dir)
        output_dir = Path(args.output_dir)
        it = tqdm(run_dir.glob("*.trec"))
        for run_path in it:
            output_path = output_dir / f"{run_path.stem}.csv"

            run = pt.io.read_results(run_path.absolute().as_posix())
            results = eval_one(run, qrels)

            pl.DataFrame(results).write_csv(output_path)

        print(f"Eval complete! Ouput: {output_dir}")

        return

    if args.run_path and args.output_path:
        run_path = Path(args.run_path)
        output_path = Path(args.output_path)

        run = pt.io.read_results(run_path.absolute().as_posix())
        results = eval_one(run, qrels)

        pl.DataFrame(results).write_csv(output_path)

        print(f"Eval complete. Ouput: {output_path}")

        return

    raise ValueError("Provide directory or file path, no mix in between.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="dataset/baseline")
    ap.add_argument("--run_dir")
    ap.add_argument("--run_path")
    ap.add_argument("--output_dir")
    ap.add_argument("--output_path")
    args = ap.parse_args()

    print(args)

    main(args)

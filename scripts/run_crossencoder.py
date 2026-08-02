import argparse
from argparse import Namespace
from pathlib import Path

import polars as pl
from constants import TOPK
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm
from utils import seed_everything


def main(args: Namespace):
    seed_everything(args.seed)

    output_path = Path(args.output_path)

    input_dir = Path(args.input_dir)
    queries_path = input_dir / "queries.csv"
    corpus_path = input_dir / "corpus.csv"

    docs = pl.read_csv(corpus_path, encoding="utf8").select(["docno", "text"])
    topics = pl.read_csv(queries_path, encoding="utf8").select(["qid", "query"])

    print(f"Init retriever: {args.model_path}")
    model = CrossEncoder(args.model_path, num_labels=1)

    def predict_pairs(pairs):
        return model.predict(
            pairs,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=args.verbose,
        )

    docnos = docs["docno"].to_list()
    texts = docs["text"].to_list()

    it = topics.iter_rows(named=True)
    if args.verbose:
        it = tqdm(it, total=len(topics))

    results = []
    for row in it:
        qid, query = row["qid"], row["query"]
        pairs = [[query, text] for text in texts]
        scores = predict_pairs(pairs)
        order = scores.argsort()[::-1][:TOPK]
        for rank, idx in enumerate(order):
            results.append(
                (qid, "Q0", docnos[idx], rank, float(scores[idx]), "CE")
            )

    pl.DataFrame(
        results, schema=["qid", "Q0", "docno", "rank", "score", "tag"]
    ).write_csv(output_path, separator=" ", include_header=False)

    print(f"Run complete. Ouput: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path", default="models/mmbert-amsunda-crossencoder/final"
    )
    ap.add_argument("--input_dir", default="dataset/baseline")
    ap.add_argument("--output_path", default="runs/crossencoder.trec")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", default=42)
    args = ap.parse_args()

    main(args)

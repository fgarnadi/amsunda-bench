import argparse
import logging
import os
from argparse import Namespace
from pathlib import Path

import polars as pl
import yaml
from datasets import Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    losses,
)
from utils import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cross-encoder")


def main(args: Namespace):
    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed: int = cfg.get("seed", 42)
    seed_everything(seed)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(cfg["data_path"])
    df = pl.read_csv(data_path, columns=["query", "positive", "negative"])

    # For BCELoss
    # https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss
    queries = df["query"].to_list()
    positives = df["positive"].to_list()
    negatives = df["negative"].to_list()
    train_ds = Dataset.from_dict(
        {
            "sentence1": queries + queries,
            "sentence2": positives + negatives,
            "label": [1] * len(positives) + [0] * len(negatives),
        }
    )

    logger.info(f"Loaded {len(df)} rows from {data_path.name}")

    # model
    base_model = cfg["base_model"]
    use_bf16 = bool(cfg.get("bf16", False))
    use_fp16 = bool(cfg.get("fp16", False))
    gc_on = bool(cfg.get("gradient_checkpointing", False))

    logger.info(f"Loading model for: {base_model}")
    model = CrossEncoder(base_model, num_labels=1)
    loss = losses.BinaryCrossEntropyLoss(model=model)

    # training
    logger.info("Training preparation...")

    train_batch_size = int(cfg.get("per_device_train_batch_size", 8))
    num_train_epochs = int(cfg.get("num_train_epochs", 10))
    max_steps = int(cfg.get("max_steps", -1))
    train_args = CrossEncoderTrainingArguments(
        output_dir=output_dir.absolute().as_posix(),
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=int(
            cfg.get("gradient_accumulation_steps", 4)
        ),
        gradient_checkpointing=gc_on,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=cfg.get("optim", "adamw_torch_fused"),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=float(cfg.get("learning_rate", 8e-4)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.06)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        logging_strategy=cfg.get("logging_strategy", "epoch"),
        logging_steps=int(cfg.get("logging_steps", 50)),
        save_strategy=cfg.get("save_strategy", "epoch"),
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(cfg.get("load_best_model_at_end", False)),
        report_to="none",
        seed=seed,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        loss=loss,
    )

    logger.info("Start training...")
    trainer.train()

    # save final
    final_dir = output_dir / "final"
    trainer.save_model(final_dir.absolute().as_posix())

    logger.info(f"Finished training, saved to {final_dir.as_posix()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cross-encoder.yaml")
    args = ap.parse_args()

    main(args)

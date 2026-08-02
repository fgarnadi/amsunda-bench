import argparse
import logging
import multiprocessing as mp
import os
from argparse import Namespace
from itertools import chain
from pathlib import Path

import polars as pl
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from utils import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("dapt")


def main(args: Namespace):
    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed: int = cfg.get("seed", 42)
    seed_everything(seed)

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    corpus_path = Path(cfg["corpus_path"])
    text_col = cfg.get("text_column", "text")

    eval_fraction = cfg.get("eval_fraction", 0.0)
    eval_strategy = cfg.get("eval_strategy", "no")

    df = (
        pl.read_csv(corpus_path, columns=[text_col])
        .rename({text_col: "text"})
        .drop_nulls(subset=["text"])
        .sample(fraction=1.0, seed=seed)  # shuffle
    )

    logger.info(
        f"Loaded {len(df)} documents from {corpus_path.name} with eval fraction {eval_fraction}"
    )
    if eval_strategy != "no" and eval_fraction > 0:
        n_val = int(len(df) * eval_fraction)
        eval_df = df.tail(n_val)
        train_df = df.head(len(df) - n_val)
    else:
        train_df, eval_df = df, None

    train_ds = Dataset.from_pandas(train_df.to_pandas())
    if eval_df is not None:
        eval_ds = Dataset.from_pandas(eval_df.to_pandas())

    # model
    base_model = cfg["base_model"]
    use_bf16 = bool(cfg.get("bf16", False))
    use_fp16 = bool(cfg.get("fp16", False))
    gc_on = bool(cfg.get("gradient_checkpointing", False))

    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else None

    logger.info(f"Loading model and tokenizer for: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(base_model, dtype=dtype)

    use_lora = bool(cfg.get("use_lora", True))
    logger.info(f"Using LoRA: {use_lora}")
    if use_lora:
        peft_config = LoraConfig(
            bias="none",
            r=int(cfg.get("lora_rank", 32)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=cfg.get("lora_modules", []),
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # tokenize
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=False,
            return_special_tokens_mask=True,
        )

    block_size = int(cfg.get("block_size", 512))

    # cap num_proc to available CPUs
    requested_num_proc = int(cfg.get("num_proc", 4))
    num_proc = min(requested_num_proc, max(1, mp.cpu_count() - 1))

    logger.info("Transforming dataset: tokenizing")
    train_ds = train_ds.map(
        tokenize, batched=True, num_proc=num_proc, remove_columns=["text"]
    )
    eval_ds = (
        eval_ds.map(
            tokenize, batched=True, num_proc=num_proc, remove_columns=["text"]
        )
        if eval_df is not None
        else None
    )

    def group_texts(examples):
        concatenated = {
            k: list(chain.from_iterable(examples[k])) for k in examples.keys()
        }

        total_length = len(concatenated["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        else:
            # If total text is less than block size, return empty to drop it
            return {k: [] for k in concatenated}

        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated.items()
        }

        return result

    logger.info("Transforming dataset: grouping")
    train_ds = train_ds.map(group_texts, batched=True, num_proc=num_proc)
    if eval_ds is not None:
        eval_ds = eval_ds.map(group_texts, batched=True, num_proc=num_proc)

    # data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(cfg.get("mlm_probability", 0.25)),
    )

    # training
    logger.info("Training preparation...")

    train_batch_size = int(cfg.get("per_device_train_batch_size", 8))
    num_train_epochs = int(cfg.get("num_train_epochs", 50))
    max_steps = int(cfg.get("max_steps", -1))
    train_args = TrainingArguments(
        output_dir=output_dir.absolute().as_posix(),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=int(
            cfg.get("per_device_eval_batch_size", train_batch_size)
        ),
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
        eval_steps=int(cfg.get("eval_steps", 200)),
        eval_strategy=eval_strategy,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    logger.info("Start training...")
    trainer.train()

    # save final
    final_dir = output_dir / "final"
    trainer.save_model(final_dir.absolute().as_posix())
    tokenizer.save_pretrained(final_dir)

    logger.info(f"Finished training, saved to {final_dir.as_posix()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dapt.yaml")
    args = ap.parse_args()

    main(args)

.PHONY: help sync sync-update check fmt transform-raw
.DEFAULT_GOAL := help

help: ## display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

sync: ## sync dependencies of all workspace
	@uv sync --all-packages

sync-update: ## sync dependencies of all workspace with update
	@uv sync -U --all-packages

check: ## check for lint and format errors
	@ruff check || true
	@ruff format --check || true

fmt: ## fix lint and format 
	@ruff check --fix
	@ruff format

# ----- Dataset Operation

transform-raw:
	uv run scripts/transform_raw_dataset.py \
		--input_dir dataset/raw \
		--out_train_dir dataset/training \
		--out_baseline_dir dataset/baseline

pretokenize-2gram:
	uv run scripts/ngram_pretokenize.py \
		--input_dir dataset/baseline \
		--out_dir   dataset/baseline_ng2 \
		--ngram 2


# ----- BM25

## ---- Index

index-bm25:
	uv run scripts/index_bm25.py \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25

index-bm25-ng2:
	uv run scripts/index_bm25.py \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25.ng2 \
		--ngram 2

## ---- Retrieve

generate-bm25-run:
	uv run scripts/run_bm25.py \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25 \
		--output_path runs/bm25.trec \
		--verbose

generate-bm25-ng2-run:
	uv run scripts/run_bm25.py \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25.ng2 \
		--output_path runs/bm25.ng2.trec \
		--ngram 2 \
		--verbose

# ----- TF-IDF

## ---- Retrieve

generate-tfidf-run:
	uv run scripts/run_tfidf.py \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25 \
		--output_path runs/tfidf.trec \
		--verbose

# ----- DAPT

train-mmbert-dapt:
	uv run scripts/train_dapt.py \
		--config configs/mmbert-dapt.yaml

train-mbert-dapt:
	uv run scripts/train_dapt.py \
		--config configs/mbert-dapt.yaml

train-xlmr-dapt:
	uv run scripts/train_dapt.py \
		--config configs/xlmr-dapt.yaml

# ----- Bi-Encoder

## ---- Training

train-mmbert-biencoder:
	uv run scripts/train_biencoder.py \
		--config configs/mmbert-biencoder.yaml

train-mbert-biencoder:
	uv run scripts/train_biencoder.py \
		--config configs/mbert-biencoder.yaml

train-xlmr-biencoder:
	uv run scripts/train_biencoder.py \
		--config configs/xlmr-biencoder.yaml

train-mmbert-base-biencoder:
	uv run scripts/train_biencoder.py \
		--config configs/mmbert-base-biencoder.yaml

## ---- Index

index-mmbert-biencoder:
	uv run scripts/index_biencoder.py \
		--model_path models/mmbert-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/mmbert.amsunda.biencoder.flex \
		--verbose

index-mbert-biencoder:
	uv run scripts/index_biencoder.py \
		--model_path models/mbert-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/mbert.amsunda.biencoder.flex \
		--verbose

index-xlmr-biencoder:
	uv run scripts/index_biencoder.py \
		--model_path models/xlmr-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/xlmr.amsunda.biencoder.flex \
		--verbose

index-mmbert-base-biencoder:
	uv run scripts/index_biencoder.py \
		--model_path models/mmbert-base-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/mmbert.base.amsunda.biencoder.flex \
		--verbose

## ---- Retrieve

generate-mmbert-biencoder-run:
	uv run scripts/run_biencoder.py \
		--model_path models/mmbert-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/mmbert.amsunda.biencoder.flex \
		--output_path runs/mmbert.biencoder.trec \
		--verbose

generate-mbert-biencoder-run:
	uv run scripts/run_biencoder.py \
		--model_path models/mbert-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/mbert.amsunda.biencoder.flex \
		--output_path runs/mbert.biencoder.trec \
		--verbose

generate-xlmr-biencoder-run:
	uv run scripts/run_biencoder.py \
		--model_path models/xlmr-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/xlmr.amsunda.biencoder.flex \
		--output_path runs/xlmr.biencoder.trec \
		--verbose

generate-mmbert-base-biencoder-run:
	uv run scripts/run_biencoder.py \
		--model_path models/mmbert-base-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/mmbert.base.amsunda.biencoder.flex \
		--output_path runs/mmbert.base.biencoder.trec \
		--verbose

# ----- Cross-Encoder

## ---- Training

train-mmbert-crossencoder:
	uv run scripts/train_crossencoder.py \
		--config configs/mmbert-crossencoder.yaml

train-mbert-crossencoder:
	uv run scripts/train_crossencoder.py \
		--config configs/mbert-crossencoder.yaml

train-xlmr-crossencoder:
	uv run scripts/train_crossencoder.py \
		--config configs/xlmr-crossencoder.yaml

train-mmbert-base-crossencoder:
	uv run scripts/train_crossencoder.py \
		--config configs/mmbert-base-crossencoder.yaml

## ---- Retrieve

generate-mmbert-crossencoder-run:
	uv run scripts/run_crossencoder.py \
		--model_path models/mmbert-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--output_path runs/mmbert.crossencoder.trec \
		--verbose

generate-mbert-crossencoder-run:
	uv run scripts/run_crossencoder.py \
		--model_path models/mbert-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--output_path runs/mbert.crossencoder.trec \
		--verbose

generate-xlmr-crossencoder-run:
	uv run scripts/run_crossencoder.py \
		--model_path models/xlmr-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--output_path runs/xlmr.crossencoder.trec \
		--verbose

generate-mmbert-base-crossencoder-run:
	uv run scripts/run_crossencoder.py \
		--model_path models/mmbert-base-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--output_path runs/mmbert.base.crossencoder.trec \
		--verbose

# ----- Rerank Run

## ---- mmBERT

rerank-mmbert-bm25-biencoder:
	uv run scripts/rerank_bm25_biencoder.py \
		--reranker_model_path models/mmbert-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25 \
		--reranker_index_dir indexes/terrier/mmbert.amsunda.biencoder.flex \
		--output_path runs/mmbert.bm25.biencoder.trec \
		--verbose

rerank-mmbert-bm25-crossencoder:
	uv run scripts/rerank_bm25_crossencoder.py \
		--reranker_model_path models/mmbert-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25 \
		--output_path runs/mmbert.bm25.crossencoder.trec \
		--verbose

rerank-mmbert-biencoder-crossencoder:
	uv run scripts/rerank_biencoder_crossencoder.py \
		--model_path models/mmbert-amsunda-biencoder/final \
		--reranker_model_path models/mmbert-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/mmbert.amsunda.biencoder.flex \
		--output_path runs/mmbert.biencoder.crossencoder.trec \
		--verbose

## ---- ModernBERT

rerank-mbert-bm25-biencoder:
	uv run scripts/rerank_bm25_biencoder.py \
		--reranker_model_path models/mbert-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25 \
		--reranker_index_dir indexes/terrier/mbert.amsunda.biencoder.flex \
		--output_path runs/mbert.bm25.biencoder.trec \
		--verbose

rerank-mbert-bm25-crossencoder:
	uv run scripts/rerank_bm25_crossencoder.py \
		--reranker_model_path models/mbert-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25 \
		--output_path runs/mbert.bm25.crossencoder.trec \
		--verbose

rerank-mbert-biencoder-crossencoder:
	uv run scripts/rerank_biencoder_crossencoder.py \
		--model_path models/mbert-amsunda-biencoder/final \
		--reranker_model_path models/mbert-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/mbert.amsunda.biencoder.flex \
		--output_path runs/mbert.biencoder.crossencoder.trec \
		--verbose

## ---- XLM-Roberta

rerank-xlmr-bm25-biencoder:
	uv run scripts/rerank_bm25_biencoder.py \
		--reranker_model_path models/xlmr-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25 \
		--reranker_index_dir indexes/terrier/xlmr.amsunda.biencoder.flex \
		--output_path runs/xlmr.bm25.biencoder.trec \
		--verbose

rerank-xlmr-bm25-crossencoder:
	uv run scripts/rerank_bm25_crossencoder.py \
		--reranker_model_path models/xlmr-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25 \
		--output_path runs/xlmr.bm25.crossencoder.trec \
		--verbose

rerank-xlmr-biencoder-crossencoder:
	uv run scripts/rerank_biencoder_crossencoder.py \
		--model_path models/xlmr-amsunda-biencoder/final \
		--reranker_model_path models/xlmr-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/xlmr.amsunda.biencoder.flex \
		--output_path runs/xlmr.biencoder.crossencoder.trec \
		--verbose

## ---- mmBERT no-DAPT

rerank-mmbert-base-bm25-biencoder:
	uv run scripts/rerank_bm25_biencoder.py \
		--reranker_model_path models/mmbert-base-amsunda-biencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25 \
		--reranker_index_dir indexes/terrier/mmbert.base.amsunda.biencoder.flex \
		--output_path runs/mmbert.base.bm25.biencoder.trec \
		--verbose

rerank-mmbert-base-bm25-crossencoder:
	uv run scripts/rerank_bm25_crossencoder.py \
		--reranker_model_path models/mmbert-base-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/amsunda.bm25 \
		--output_path runs/mmbert.base.bm25.crossencoder.trec \
		--verbose

rerank-mmbert-base-biencoder-crossencoder:
	uv run scripts/rerank_biencoder_crossencoder.py \
		--model_path models/mmbert-base-amsunda-biencoder/final \
		--reranker_model_path models/mmbert.base-amsunda-crossencoder/final \
		--input_dir dataset/baseline \
		--index_dir indexes/terrier/mmbert.base.amsunda.biencoder.flex \
		--output_path runs/mmbert.base.biencoder.crossencoder.trec \
		--verbose

# ---- Evaluate

evaluate-run:
	uv run scripts/eval_run.py \
		--input_dir dataset/baseline \
		--run_dir runs \
		--output_dir results

index-biencoder: index-mmbert-biencoder index-mbert-biencoder index-xlmr-biencoder index-mmbert-base-biencoder
run-biencoder: generate-mmbert-biencoder-run generate-mbert-biencoder-run generate-xlmr-biencoder-run generate-mmbert-base-biencoder-run
full-biencoder: index-biencoder run-biencoder evaluate-run
run-rerank-bm25-biencoder: rerank-mmbert-bm25-biencoder rerank-mbert-bm25-biencoder rerank-xlmr-bm25-biencoder rerank-mmbert-base-bm25-biencoder
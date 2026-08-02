#!/bin/bash

# Cookbook for running the experiments
# Basic flow:
# 0. Prepare the dataset
# 1. Train the models, skipped for lexical retrieval
# 2. Run indexing
# 3. Generate runs
# 4. Evaluate the runs
#
# Refer to the Makefile for more details on each step

# Dataset
make transform-raw # to transform the raw dataset into specified format
make pretokenize-2gram # to pretokenize the dataset into 2-gram format

# BM25

## ngram 1
make index-bm25
make generate-bm25-run

## ngram 2
make index-bm25-ng2
make generate-bm25-ng2-run

# TF-IDF

# using index-bm25 as the index for tf-idf retrieval
make generate-tfidf-run

# TRAINING

## dapt
make train-mmbert-dapt
make train-mbert-dapt
make train-xlmr-dapt

## biencoder
make train-mmbert-biencoder
make train-mbert-biencoder
make train-xlmr-biencoder
make train-mmbert-base-biencoder

## crossencoder
make train-mmbert-crossencoder
make train-mbert-crossencoder
make train-xlmr-crossencoder
make train-mmbert-base-crossencoder

# INDEX

make index-mmbert-biencoder
make index-mbert-biencoder
make index-xlmr-biencoder
make index-mmbert-base-biencoder

# RUN

## biencoder
make generate-mmbert-biencoder-run
make generate-mbert-biencoder-run
make generate-xlmr-biencoder-run
make generate-mmbert-base-biencoder-run

## crossencoder
make generate-mmbert-crossencoder-run
make generate-mbert-crossencoder-run
make generate-xlmr-crossencoder-run
make generate-mmbert-base-crossencoder-run

## rerank
make rerank-mmbert-bm25-biencoder
make rerank-mmbert-bm25-crossencoder
make rerank-mmbert-biencoder-crossencoder

make rerank-mbert-bm25-biencoder
make rerank-mbert-bm25-crossencoder
make rerank-mbert-biencoder-crossencoder

make rerank-xlmr-bm25-biencoder
make rerank-xlmr-bm25-crossencoder
make rerank-xlmr-biencoder-crossencoder

make rerank-mmbert-base-bm25-biencoder
make rerank-mmbert-base-bm25-crossencoder
make rerank-mmbert-base-biencoder-crossencoder

# EVALUATE

make evaluate-run

# Benchmarking AMSunda

This is the repository for [Benchmarking AMSunda: Evaluating Sparse and Dense Retrieval Architectures for Sundanese Information Retrieval](https://doi.org/10.1109/SIML69834.2026.11621606). The paper was published in the 2026 International Conference on Smart Computing, IoT, and Machine Learning (SIML).

This paper was using the [AMSunda dataset](https://doi.org/10.1016/j.dib.2025.111796) for its experiments.

## Project Structure

- `configs/`: Configuration files for the experiments.
- `dataset/`: Directory for the dataset.
- `notebooks/`: Jupyter notebooks for data exploration and analysis.
- `scripts/`: Python scripts for experimentation.

There are also `Makefile` and `run.sh` for reference on how to run the experiments.

## Prerequisites

### Dataset

The dataset can be obtained from Zenodo [here](https://doi.org/10.5281/zenodo.15494944).
It contains 4 files:

- `corpus.csv`: The corpus of Sundanese documents.
- `queries.csv`: The queries for information retrieval.
- `qrels.csv`: The relevance judgments for the queries.
- `triplet.csv`: The triplet data containing (query, positive document, negative document).

Place the downloaded files in the `dataset/raw/` directory.

### Installation

The project is managed using [uv](https://github.com/astral-sh/uv).  
Install the dependencies using the following command:

```bash
uv sync
```

Then refer to the `Makefile` or `run.sh` for running the experiments.

## Results

The following table shows the main results of the experiments conducted in the paper.

| Scenario and Method | nDCG@1 | nDCG@5 | nDCG@10 | MRR@10 | R@1 | R@5 | R@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lexical Baselines** | | | | | | | |
| TF-IDF | **0.5360** | **0.6388** | **0.6583** | **0.6186** | **0.5360** | **0.7259** | **0.7856** |
| BM25 | 0.5162 | 0.6253 | 0.6451 | 0.6036 | 0.5162 | 0.7175 | 0.7780 |
| **Base Scenario: mmBERT (DAPT)** | | | | | | | |
| Bi-Encoder | 0.0012 | 0.0029 | 0.0044 | 0.0029 | 0.0012 | 0.0047 | 0.0092 |
| BM25 + Bi-Encoder | 0.0845 | 0.2334 | 0.3575 | 0.2329 | 0.0845 | 0.3898 | 0.7781 |
| BM25 + Cross-Encoder | 0.0877 | 0.2569 | 0.3679 | 0.2475 | 0.0877 | 0.4312 | 0.7781 |
| Bi-Encoder + Cross-Encoder | 0.0004 | 0.0021 | 0.0038 | 0.0022 | 0.0004 | 0.0040 | 0.0092 |
| **Ablation 1: mmBERT (No DAPT)** | | | | | | | |
| Bi-Encoder | 0.0013 | 0.0027 | 0.0039 | 0.0027 | 0.0013 | 0.0041 | 0.0079 |
| BM25 + Bi-Encoder | 0.0844 | 0.2417 | 0.3618 | 0.2382 | 0.0844 | 0.4023 | 0.7781 |
| BM25 + Cross-Encoder | 0.0912 | 0.2458 | 0.3641 | 0.2423 | 0.0912 | 0.4080 | 0.7781 |
| Bi-Encoder + Cross-Encoder | 0.0007 | 0.0027 | 0.0037 | 0.0024 | 0.0007 | 0.0047 | 0.0079 |
| **Ablation 2: ModernBERT (Monolingual)** | | | | | | | |
| Bi-Encoder | 0.0368 | 0.0592 | 0.0690 | 0.0565 | 0.0368 | 0.0794 | 0.1100 |
| BM25 + Bi-Encoder | 0.1922 | 0.3549 | 0.4388 | 0.3361 | 0.1922 | 0.5161 | 0.7781 |
| BM25 + Cross-Encoder | 0.2706 | 0.4559 | 0.5053 | 0.4201 | 0.2706 | 0.6254 | 0.7781 |
| Bi-Encoder + Cross-Encoder | 0.0104 | 0.0289 | 0.0483 | 0.0302 | 0.0104 | 0.0486 | 0.1100 |
| **Ablation 3: XLM-Roberta** | | | | | | | |
| Bi-Encoder | 0.0008 | 0.0026 | 0.0035 | 0.0021 | 0.0008 | 0.0044 | 0.0072 |
| BM25 + Bi-Encoder | 0.0643 | 0.2234 | 0.3481 | 0.2198 | 0.0643 | 0.3903 | 0.7781 |
| BM25 + Cross-Encoder | 0.0804 | 0.2331 | 0.3553 | 0.2298 | 0.0804 | 0.3951 | 0.7781 |
| Bi-Encoder + Cross-Encoder | 0.0008 | 0.0024 | 0.0031 | 0.0021 | 0.0008 | 0.0040 | 0.0065 |

## Citation

Please cite our paper if you use this repository in your research:

```BibTeX
@inproceedings{11621606,
  author={Garnadi, Fajar and Alfina, Ika and Yulianti, Evi and Budi, Indra},
  booktitle={2026 International Conference on Smart Computing, IoT, and Machine Learning (SIML)}, 
  title={Benchmarking AMSunda: Evaluating Sparse and Dense Retrieval Architectures for Sundanese Information Retrieval}, 
  year={2026},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/SIML69834.2026.11621606}
}
```

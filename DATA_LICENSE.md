# Dataset terms

This repository does **not** contain the WM-811K dataset. `scripts/fetch_dataset.py` downloads
it to `data/raw/` (git-ignored) and verifies its SHA-256.

- **Dataset:** WM-811K (`LSWMD.pkl`), 811,457 wafer maps from real fab production, 172,950 of them
  labelled with one of nine failure-pattern classes.
- **Original publisher:** MIR Lab, National Taiwan University.
- **Citation (required when using the data):**
  M.-J. Wu, J.-S. R. Jang and J.-L. Chen, "Wafer Map Failure Pattern Recognition and Similarity
  Ranking for Large-Scale Data Sets," *IEEE Transactions on Semiconductor Manufacturing*, vol. 28,
  no. 1, pp. 1-12, Feb. 2015.
- **Mirror used:** <https://huggingface.co/datasets/lslattery/wafer-defect-detection> (ungated;
  the same file is distributed on Kaggle as "WM-811K wafer map").
- **Licence:** neither the mirror nor the Kaggle listing states an explicit licence. The data is
  used here for non-commercial research / portfolio purposes with attribution, and is
  not redistributed. If you plan any commercial use, clear the terms with the original publisher first.

Derived artefacts committed here are **aggregate statistics, plots and trained model weights** only.
No raw or per-wafer data is committed.

The code in this repository is MIT-licensed (see `LICENSE`).

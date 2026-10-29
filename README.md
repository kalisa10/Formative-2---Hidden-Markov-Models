# Hidden Markov Models: Human Activity Recognition

Individual project by **kalisa10**. A Hidden Markov Model, written from scratch with NumPy, infers four hidden activity states (standing, walking, jumping, still) from smartphone accelerometer and gyroscope recordings.

## Deliverables

| Item | Location |
|---|---|
| Report | [report/Hidden_Markov_Models_Report.pdf](report/Hidden_Markov_Models_Report.pdf) (source: [report/report.md](report/report.md)) |
| Notebook | [notebooks/hmm_activity_recognition.ipynb](notebooks/hmm_activity_recognition.ipynb) |
| Raw Sensor Logger recordings | `data/raw/` (training) and `data/test/` (unseen) |
| Resampled CSV files (50 Hz, with timestamp column) | `data/processed/` |
| Figures and metrics tables | `report/figures/` |

## Videos (data collection, whole body and face visible)

| Activity | Link |
|---|---|
| Standing | https://drive.google.com/file/d/1IsrgzqzaveB-ls6wlCweYWEGJ0Nyu8lj/view?usp=sharing |
| Walking | https://drive.google.com/file/d/1ZM3M0xJ0T_1srJwOV-AU4O60Caq5uFqG/view?usp=sharing |
| Jumping | https://drive.google.com/file/d/16drKydAr9-TuD7tQV7_84wIcj8jVovsL/view?usp=sharing |
| Still | https://drive.google.com/file/d/1eaPmUTchmALTQtLwEn2yG-X0EFtxfR40/view?usp=sharing |

## Data

* Device: Google Pixel 3a, Sensor Logger app, accelerometer and gyroscope. Native rate 50.7 Hz, harmonised to 50 Hz by linear interpolation.
* Training: 10 recordings per activity (40 in total), recorded on 26 and 29 Oct 2026 in several places and phone positions (about 118 s per activity). Unseen test set: 2 recordings per activity (8 in total), never used for training, scaling or PCA.
* Names: `<member>_<activity>_<nn>` (for example `kalisa10_walking_03`).
* `data/recordings_manifest.csv` and `data/test_manifest.csv` map each file to its original Sensor Logger name and recording time.

## Method

1. Load, trim the start/end taps, resample every channel to 50 Hz.
2. Window: 2 s (100 samples), 50 % overlap.
3. Features (38 per window): time domain (mean, variance, standard deviation, SMA, axis correlation) and frequency domain (FFT dominant frequency, spectral energy, four band energies).
4. Z-score standardisation and PCA to 8 components, fitted on training windows only.
5. HMM with full-covariance Gaussian emissions: supervised initialisation, **Baum-Welch** (stops when the relative log-likelihood gain is below 1e-4), **Viterbi** decoding, forward-backward posteriors.
6. Evaluation on the unseen recordings: confusion matrix, sensitivity, specificity, accuracy, transition and emission plots.

## Results (unseen data, Viterbi)

| Activity | Windows | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Standing | 15 | 1.000 | 1.000 | 1.000 |
| Walking | 17 | 1.000 | 1.000 | 1.000 |
| Jumping | 16 | 1.000 | 1.000 | 1.000 |
| Still | 16 | 1.000 | 1.000 | 1.000 |
| All | 64 | | | 1.000 |

The unseen clips come from the same person and phone and there are only 64 windows, so the score is optimistic. See the report for the discussion.

## How to run

Requirements: Python 3.10+, `numpy`, `pandas`, `matplotlib`, `scipy`, `jupyter`.

```
pip install numpy pandas matplotlib scipy jupyter
cd notebooks
jupyter notebook hmm_activity_recognition.ipynb
```

Run all cells from the top. The notebook reads `../data/raw` and `../data/test`, writes the resampled files to `../data/processed` and the figures to `../report/figures`. Training happens in section 5 (supervised fit followed by Baum-Welch). Settings are at the top: `WIN_SEC`, `OVERLAP`, `N_PCA`, `BW_TOL`, `BW_MAX_ITER`, `SEED`.

To use your own recordings, unzip the Sensor Logger exports into `data/raw/` (training) and `data/test/` (unseen), named `<member>_<activity>_<nn>`.

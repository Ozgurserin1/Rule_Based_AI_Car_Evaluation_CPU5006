# Car Acceptability: Flat and Hierarchical Rule-Based Classification

CPU5006 — Assessment 1 | Bath Spa University

## Research question

How do flat and hierarchical rule-based classifiers compare when predicting acceptability labels in the UCI Car Evaluation Dataset?

## Dataset and methods

The project uses 1,728 categorical records with six input features and four target classes (`unacc`, `acc`, `good`, `vgood`). It compares an ordered sequential-covering rule list with a staged classifier using PRICE, COMFORT and TECH intermediate decisions.

A stratified 20% final holdout (346 records; random seed 42) is reserved for evaluation. The primary performance measure is macro-F1. Five additional paired splits are exploratory and should not be treated as independent confirmatory experiments.

## Final holdout results

| Measure | Flat rules | Hierarchical rules |
|---|---:|---:|
| Accuracy | 0.8786 | 0.8873 |
| Macro-F1 | 0.7323 | 0.7652 |
| Correct cases | 304 / 346 | 307 / 346 |

Although the hierarchical classifier performed better on the fixed holdout, the flat classifier had higher macro-F1 in four of five additional exploratory paired splits. The results do not establish general superiority for either design.

![Experimental classification results](Experimental%20Results_%20Rules%20and%20Confusion%20Matrices%20%281%29.png)

## Project structure

- `src/` — classification models and evaluation entry point
- `data/car.data` — dataset used in the experiments
- `tests/` — model tests
- `outputs/` — saved results and supporting metrics
- `figures/` — research figures
- `METHODOLOGY_FROZEN.json` — recorded experiment settings
- `FILE_CHECKSUMS.txt` — file fingerprints for integrity checks

## Setup and verification

Install Python 3.10+ and the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the development experiment with:

```bash
python src/run_experiment.py
```

See `src/run_experiment.py --help` for supported options before running final-holdout evaluation. Existing final results are included in `outputs/`.

## Data source

Bohanec, M. (1988) *Car Evaluation* [dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JP48

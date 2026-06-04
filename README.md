# F1 Pit Prediction

Predict the probability a Formula 1 driver pits on the next lap. The data is tabular, one row per lap. The target is binary and the metric is AUC.

The final model reaches about 0.933 CV OOF AUC.

## The task

Each row is one lap for one driver. The target, PitNextLap, is 1 when the driver pits on the following lap. The data is synthetic, generated from real F1 timing. Test labels are not provided, so I report cross-validated out-of-fold AUC throughout.

## Data

- data/train.csv and data/test.csv come from the competition. Download them into data/. They are not in this repo.
- data/collected.csv is real FastF1 lap data I pulled myself. One engineered feature matches against it. It is included.

## Pipeline

Six notebooks, run in order. Each one saves what the next reads.

1. 01_EDA. Target balance, per-feature distributions, and the relationship between each feature and the target.
2. 02_Feature_Engineering. Builds the engineered features and writes the parquet the models read.
3. 03_Baselines. Logistic regression, XGBoost, LightGBM, CatBoost. Saves each model's out-of-fold and test predictions.
4. 04_Stacking. Stage-2 models on the stage-1 outputs. A residual stack, a feature stack, and a logistic meta-blender.
5. 05_Hill_Climbing. Greedy weighted blend over the base and stacking models.
6. 06_Pseudo_Label. Pseudo labeling and distillation into a single XGBoost. This writes the final submission.

## Key choices

- Validation is 10-fold StratifiedGroupKFold grouped by Race and Year. A race never spans train and validation, so the out-of-fold scores are honest.
- Target encodings get rebuilt inside each fold, never once up front. That keeps the target out of validation.
- One feature matches every synthetic lap to the nearest real FastF1 lap with a kDTree, per race. The distance doubles as a drift detector for the synthetic generator.
- Tree models average over several seeds. LightGBM is tuned with Optuna.

## Results

CV OOF AUC by model.

| model | OOF AUC |
| --- | --- |
| logistic regression | 0.879 |
| XGBoost | 0.924 |
| LightGBM | 0.930 |
| CatBoost | 0.932 |
| hill-climb blend | 0.933 |
| distilled XGB (final) | 0.933 |

## What did not help

- A neural net earned no weight in the blend, so I dropped it.
- A deeper four-level stack, with extra feature extraction and more base models and stackers, scored below this simpler pipeline. I kept the simpler one.

## Run it

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_EDA.ipynb
```

Run 01 through 06 in order, or open them in Jupyter or VS Code and use Run All.

## Stack

Python, pandas, numpy, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, matplotlib.

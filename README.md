# Real Estate Sales Data Analysis

Predicting residential sale prices in King County, Washington, using regression models built on 21,613 home sales from May 2014 to May 2015.

## Key Findings

* Living area (`sqft_living`) has the strongest correlation with price (r = 0.70), followed by King County building grade (r = 0.67).
* Waterfront homes have a median sale price roughly 3x higher than non-waterfront homes.
* Adding second-order polynomial features to a Ridge model raised test-set R² from 0.65 to 0.73.
* Cross-validation confirms the polynomial model's improvement is real, not an artifact of the train/test split: it scores higher than the linear Ridge model in all 5 folds.

## Results

|Model|Features|R²|Evaluated on|
|-|-|-|-|
|Linear regression|`sqft_living` only|0.49|Training data|
|Linear regression|13 features|0.66|Training data|
|Pipeline (scaling + polynomial + linear)|13 features|0.76|Training data|
|Ridge (α = 0.1)|13 features|0.65|Test set (15%)|
|Ridge (α = 0.1) + degree-2 polynomial|13 features|0.73|Test set (15%)|
|Ridge (α = 0.1)|13 features|0.658 ± 0.008|5-fold cross-validation|
|Ridge (α = 0.1) + degree-2 polynomial|13 features|**0.742 ± 0.021**|5-fold cross-validation|
 
![Sale price by waterfront view](images/waterfront_boxplot.png)
![Sale price vs. above-ground square footage](images/sqft_vs_price.png)

## Approach

1. **Cleaning:** dropped the ID column and converted the date column from raw string to datetime. Missing bedroom and bathroom values are filled with the column means as a safeguard, though this copy of the data has none.
2. **EDA:** looked at feature distributions, outliers, and each feature's correlation with price.
3. **Modelling:** built single-feature and multi-feature linear regressions, then a scikit-learn pipeline combining scaling and polynomial features.
4. **Refinement:** used Ridge regularization on scaled features with an 85/15 train/test split, with and without polynomial features.
5. **Validation:** compared both Ridge models with 5-fold cross-validation, using the same pipelines so preprocessing is refit inside each fold.

## Project Structure

```
├── WA-realestate.ipynb   # the analysis: EDA, modelling, and interpretation
├── train.py              # reproduces the results table from the command line
├── kc_housing/
│   ├── config.py         # paths, feature list, and model parameters
│   ├── data.py           # loading and cleaning
│   ├── models.py         # model pipelines
│   ├── evaluate.py       # scoring helpers and the results table
│   └── plots.py          # EDA charts
├── tests/                # pytest tests (run without the dataset)
└── images/               # charts used in this README
```

## Limitations and Next Steps

* Price is heavily right-skewed; a log transform of the target would likely improve fit.
* Location (zipcode, lat/long) is mostly unused; encoding neighbourhood effects is the obvious next improvement.
* The Ridge alpha was fixed at 0.1. Tuning it with a grid search could improve results.

## Data

[King County House Sales dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) (public domain). See the notebook for the full data dictionary.

## Running It

Download `kc_house_data.csv` from the Kaggle link above and put it in the repo root. Then, to reproduce the results table:

```
pip install -r requirements.txt
python train.py                 # prints the results table
python train.py --save-figures  # also regenerates the charts in images/
```

To open the notebook or run the tests:

```
pip install -r requirements-dev.txt
jupyter notebook WA-realestate.ipynb
pytest
```

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, pytest

*Completed as part of the IBM Data Science Professional Certificate.*


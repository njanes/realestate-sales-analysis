# Real Estate Sales Data Analysis

Predicting residential sale prices in King County, Washington, using regression models built on 21,613 home sales from May 2014 to May 2015.

## Key Findings

* Living area (`sqft_living`) has the strongest correlation with price (r = 0.70), followed by King County building grade (r = 0.67).
* Waterfront homes have a median sale price roughly 3x higher than non-waterfront homes.
* Adding second-order polynomial features to a Ridge model raised test-set R² from 0.65 to 0.72.
* Cross-validation confirms the polynomial model's improvement is real, not an artifact of the train/test split: it holds across all 5 folds with a low standard deviation.

## Results

|Model|Features|R²|Evaluated on|
|-|-|-|-|
|Linear regression|`sqft_living` only|0.49|Training data|
|Linear regression|11 features|0.66|Training data|
|Pipeline (scaling + polynomial + linear)|11 features|0.76|Training data|
|Ridge (α = 0.1)|11 features|0.65|Test set (15%)|
|Ridge (α = 0.1) + degree-2 polynomial|11 features|0.72|Test set (15%)|
|Ridge (α = 0.1)|11 features|0.658 ± 0.007|5-fold cross-validation|
|Ridge (α = 0.1) + degree-2 polynomial|11 features|**0.740 ± 0.021**|5-fold cross-validation|
 
![Sale price by waterfront view](images/boxplot.png)
![Sale price vs. above-ground square footage](images/scatter.png)

## Approach

1. **Cleaning:** dropped the ID column and converted the date column from raw string to datetime. The data has no blank cells, but 13 homes list 0 bedrooms and 10 list 0 bathrooms; I treated these, and one 33-bedroom typo, as missing and filled them with the column medians.
2. **EDA:** looked at feature distributions, outliers, and each feature's correlation with price.
3. **Modelling:** built single-feature and multi-feature linear regressions, then a scikit-learn pipeline combining scaling and polynomial features. For house size, the models use `sqft_living` alone: `sqft_above` and `sqft_basement` add up to it exactly, and keeping the basement split raised cross-validated R² by only 0.004.
4. **Refinement:** used Ridge regularization on standardized features with an 85/15 train/test split, with and without polynomial features.
5. **Validation:** compared both Ridge models with 5-fold cross-validation, refitting preprocessing inside each fold.

## Limitations and Next Steps

* Price is heavily right-skewed; a log transform of the target would likely improve fit.
* Location (zipcode, lat/long) is mostly unused; encoding neighbourhood effects is the most likely next improvement.
* The Ridge alpha was fixed at 0.1. Tuning it with a grid search could improve results.

## Data

[King County House Sales dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) (public domain). Save the CSV as `data/kc_house_data.csv`. See the notebook for the full data dictionary.

## Running It

```
pip install -r requirements.txt
jupyter notebook WA_realestate.ipynb
```

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn

*Completed as part of the IBM Data Science Professional Certificate.*


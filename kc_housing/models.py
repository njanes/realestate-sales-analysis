"""Model specs, defined once so every section of the analysis uses the same ones."""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import ALPHA


def poly_linear(degree=2):
    """Scaling, then polynomial features, then ordinary least squares."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", LinearRegression()),
    ])


def ridge(alpha=ALPHA, degree=1):
    """Ridge regression on scaled features, with optional polynomial terms.

    Scaling comes first so alpha penalises every coefficient on the same
    footing. Because the preprocessing lives inside the pipeline, it is only
    ever fit on training data, both on a holdout split and inside each
    cross-validation fold.
    """
    steps = [("scale", StandardScaler())]
    if degree > 1:
        steps.append(
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False))
        )
    steps.append(("model", Ridge(alpha=alpha)))
    return Pipeline(steps)

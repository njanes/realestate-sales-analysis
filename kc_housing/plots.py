"""The exploratory charts shown in the notebook and README."""

import matplotlib.pyplot as plt
import seaborn as sns

from .config import BLUE, IMAGES_DIR, NAVY, PINK, RANDOM_STATE, TARGET


def _finish(ax, save_to):
    ax.ticklabel_format(style="plain", axis="y")
    if save_to is not None:
        ax.figure.savefig(save_to, dpi=100, bbox_inches="tight")
    return ax


def plot_waterfront(df, ax=None, save_to=None):
    """Box plot of sale price for homes with and without a waterfront view."""
    ax = sns.boxplot(x="waterfront", y=TARGET, data=df, color=BLUE, ax=ax)
    ax.set(
        xlabel="Waterfront View",
        ylabel="Sale Price",
        xticks=[0, 1],
        xticklabels=["No", "Yes"],
    )
    return _finish(ax, save_to)


def plot_sqft_above(df, ax=None, save_to=None):
    """Scatter plot of sale price against above-ground square footage, with a fitted line."""
    ax = sns.regplot(
        x="sqft_above",
        y=TARGET,
        data=df,
        color=BLUE,
        scatter_kws={"s": 25, "alpha": 0.5, "edgecolors": NAVY},
        line_kws={"alpha": 0.7, "color": PINK},
        seed=RANDOM_STATE,  # fixes the bootstrap for the confidence band
        ax=ax,
    )
    ax.set(xlabel="Above-Ground Square Footage", ylabel="Sale Price ($USD)")
    return _finish(ax, save_to)


FIGURES = {
    "waterfront_boxplot.png": plot_waterfront,
    "sqft_vs_price.png": plot_sqft_above,
}


def save_figures(df, out_dir=IMAGES_DIR):
    """Regenerate every chart in the README's images folder."""
    out_dir.mkdir(exist_ok=True)
    for name, plot in FIGURES.items():
        fig, ax = plt.subplots()
        plot(df, ax=ax, save_to=out_dir / name)
        plt.close(fig)

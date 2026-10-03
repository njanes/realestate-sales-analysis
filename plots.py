import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cleaning import DATA_PATH, clean


def waterfront_boxplot(data):
    fig, ax = plt.subplots()
    sns.boxplot(x="waterfront", y="price", data=data, color="#1d7bdb", ax=ax)
    ax.set(
        xlabel="Waterfront View",
        ylabel="Sale Price ($USD)",
        xticks=[0, 1],
        xticklabels=["No", "Yes"],
    )
    ax.set_title("Waterfront homes sell for about three times as much",
                 fontsize=12, fontweight="bold", x=0.41, y=1.01, pad=20)
    ax.ticklabel_format(style="plain", axis="y")
    return fig


def sqft_scatter(data):
    fig, ax = plt.subplots()
    sns.regplot(
        x="sqft_living",
        y="price",
        data=data,
        color="#1d7bdb",
        scatter_kws={"s": 25, "alpha": 0.5, "edgecolors": "#082441"},
        line_kws={"alpha": 0.7, "color": "#f20079"},
        seed=1,
        ax=ax,
    )
    ax.set(xlabel="Square Footage", ylabel="Sale Price ($USD)")
    ax.set_title("Sale price rises with living area (r = 0.70)",
                 fontsize=12, fontweight="bold", x=0.41, y=1.01, pad=20)
    ax.ticklabel_format(style="plain", axis="y")
    return fig


if __name__ == "__main__":
    data = clean(pd.read_csv(DATA_PATH))
    waterfront_boxplot(data).savefig("images/boxplot.png", bbox_inches="tight")
    sqft_scatter(data).savefig("images/scatter.png", bbox_inches="tight")

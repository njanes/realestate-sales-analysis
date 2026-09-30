"""Shared settings for the analysis: paths, features, and model parameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "kc_house_data.csv"
IMAGES_DIR = ROOT / "images"

TARGET = "price"
FEATURES = [
    "floors",
    "waterfront",
    "lat",
    "bedrooms",
    "sqft_basement",
    "view",
    "bathrooms",
    "sqft_above",
    "grade",
    "sqft_living",
    "sqft_living15",
    "sqft_lot",
    "sqft_lot15",
]

# Evaluation
TEST_SIZE = 0.15
RANDOM_STATE = 1
N_SPLITS = 5

# Ridge regularisation strength
ALPHA = 0.1

# Plot colours
BLUE = "#1d7bdb"
NAVY = "#082441"
PINK = "#f20079"

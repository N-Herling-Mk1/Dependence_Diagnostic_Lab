"""
config.py -- edit these for your ROOT->CSV exports.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # --- inputs ---
    features_csv: str = "features.csv"      # raw NN input features (one row/event)
    scores_csv:   str = "scores.csv"        # NN1/NN2 output scores (one row/event)

    nn1_features: List[str] = field(default_factory=list)   # NN1 input columns
    nn2_features: List[str] = field(default_factory=list)   # NN2 input columns
    shared_features: List[str] = field(default_factory=list)  # for the bridge (Z)

    nn1_score: str = "nn1_score"
    nn2_score: str = "nn2_score"

    # --- ABCD ---
    cut_nn1: float = 0.5
    cut_nn2: float = 0.5

    # --- sampling / iterations ---
    dcor_subsample: int = 2000
    matrix_subsample: int = 1500
    n_perm: int = 500
    n_boot: int = 2000
    n_slices: int = 5
    seed: int = 42

    # --- output ---
    outdir: str = "helix_dep_report"
    make_plots: bool = True

    # TRON palette for plots (near-black bg, cyan/amber/orange)
    bg: str = "#05080d"
    fg: str = "#7fe9ff"
    accent: str = "#ffb347"
    hot: str = "#ff7a18"

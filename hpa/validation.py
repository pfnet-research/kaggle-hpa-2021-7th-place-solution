from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def get_folds(df: pd.DataFrame, n_folds: int) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    binary_labels = np.zeros((len(df), 19), dtype=np.bool8)
    for i, label in enumerate(df["Label"].to_numpy()):
        for j in map(int, label.split("|")):
            binary_labels[i, j] = True

    kf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    return list(kf.split(np.arange(len(df)), binary_labels))

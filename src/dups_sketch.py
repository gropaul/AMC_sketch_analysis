
from typing import List, Dict

import numpy as np
import pandas as pd
import tqdm

from src.data_generator import hash_int
from src.my_types import Relation
from src.sketch import AMSSketchPlusMinus


def get_sketch_metrics(sketch: AMSSketchPlusMinus) -> Dict[str, float]:
    original_array = np.array(sketch.array)
    # make the array in range [-1, 1]
    norm_array = original_array / np.max(np.abs(original_array))

    abs_array = np.abs(original_array)

    # make the array in range [0, 1]
    norm_abs_array = abs_array / np.max(abs_array)
    n_updates = sketch.updates
    sqrt_n_updates = np.sqrt(n_updates)
    log_n_updates = np.log(n_updates)
    return {
        'std': np.std(original_array),
        'std_norm': np.std(norm_array),
        'std_abs': np.std(abs_array),
        'std_norm_abs': np.std(norm_abs_array),
        'range': np.max(original_array) - np.min(original_array),
        'range_abs': np.max(abs_array) - np.min(abs_array),
        'mean': np.mean(original_array),
        'mean_abs': np.mean(abs_array),
        'mean_norm': np.mean(norm_array),
        'mean_norm_abs': np.mean(norm_abs_array),
        'mean_per_updates': np.mean(original_array) / n_updates,
        'mean_abs_per_updates': np.mean(abs_array) / n_updates,
        'pow_2_mean_abs_per_updates': np.mean(abs_array) / n_updates ** 2,
        'pow_3_mean_abs_per_updates': np.mean(abs_array) / n_updates ** 3,
        'pow_4_mean_abs_per_updates': np.mean(abs_array) / n_updates ** 4,
        'mean_abs_per_sqrt_updates': np.mean(abs_array) / sqrt_n_updates,
        'mean_abs_per_log_updates': np.mean(abs_array) / log_n_updates,
        'mean_norm_per_updates': np.mean(norm_array) / n_updates,
        'mean_norm_abs_per_updates': np.mean(norm_abs_array) / n_updates,
    }


def test_number_of_duplicates_with_sketch(relation: Relation, df: pd.DataFrame) -> pd.DataFrame:
    sketch = AMSSketchPlusMinus(20)
    for i, value in enumerate(relation):
        hash = hash_int(value)
        sketch.update(hash, 1)

    metrics_sketch = get_sketch_metrics(sketch)
    metrics_data = {
        'n_rows': len(relation),
        'n_duplicates': len(relation) / len(set(relation))
    }

    # Create a DataFrame for the current relation's data and metrics
    new_row = pd.DataFrame([{**metrics_data, **metrics_sketch}])
    # Concatenate with the main DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    return df


def run_experiment_dups(data_space: List[Relation]) -> pd.DataFrame:
    df = pd.DataFrame()

    for data in tqdm.tqdm(data_space):
        df = test_number_of_duplicates_with_sketch(data, df)

    return df

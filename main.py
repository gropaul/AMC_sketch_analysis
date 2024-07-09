import itertools
import random
from math import floor
from typing import List, Dict
import pandas as pd
import numpy as np
import tqdm

from sketch import AMSSketchSimple

Relation = List[int]


def get_relation(n_rows: int, n_duplicates: float) -> Relation:
    value_range = n_rows // n_duplicates
    data = []
    for i in range(n_rows):
        random_value = random.randint(0, floor(value_range))
        data.append(random_value)

    return data


def get_data_space(n_rows: List[int], n_duplicates: List[float]) -> List[Relation]:
    data_space = []

    for n_row, n_duplicate in itertools.product(n_rows, n_duplicates):
        data = get_relation(n_row, n_duplicate)
        data_space.append(data)

    return data_space


def hash_int(value: int) -> int:
    return hash(value)


def get_sketch_metrics(sketch: AMSSketchSimple) -> Dict[str, float]:
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


def test_relation(relation: Relation, df: pd.DataFrame) -> pd.DataFrame:
    sketch = AMSSketchSimple(20)
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


def run_experiment(data_space: List[Relation]) -> pd.DataFrame:
    df = pd.DataFrame()

    for data in tqdm.tqdm(data_space):
        df = test_relation(data, df)

    return df


def main():
    # set the seed
    seed = 42
    random.seed(seed)

    n_rows = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    n_duplicates = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

    data_space = get_data_space(n_rows, n_duplicates)
    df = run_experiment(data_space)
    # save as csv
    df.to_csv('sketch_results.csv', index=False)


if __name__ == '__main__':
    main()

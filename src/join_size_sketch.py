import random
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from src.data_generator import hash_int
from src.my_types import Relation
from src.sketch import AMSSketchCounter, AMSSketchPlusMinus


def join(a: Relation, b: Relation) -> Relation:
    """
    Joins two relations a and b on the common join key
    """

    # build a hash table for the first relation
    hash_table = {}
    for value in a:
        if value not in hash_table:
            hash_table[value] = 1
        else:
            hash_table[value] += 1

    # join the second relation
    result: Relation = []

    for value in b:
        if value in hash_table:
            matches = hash_table[value]
            for _ in range(matches):
                result.append(value)

    return result


def get_join_input(possible_relations: List[Relation], n_relations: int) -> List[Tuple[Relation, Relation]]:
    result = []
    for i in range(n_relations):
        random_a = random.choice(possible_relations)
        random_b = random.choice(possible_relations)

        result.append((random_a, random_b))

    return result


def get_join_estimate(a: Relation, b: Relation) -> int:
    """
    Estimates the size of the join of two relations a and b
    """
    # get the size of the relations
    size_a = len(a)
    size_b = len(b)

    # get the number of duplicates in the relations
    duplicates_a = size_a - len(set(a))
    duplicates_b = size_b - len(set(b))

    # get the number of duplicates in the join
    duplicates_join = duplicates_a + duplicates_b

    # get the size of the join
    size_join = size_a + size_b - duplicates_join

    return size_join


def median(values: List[int]) -> int:
    values.sort()
    n = len(values)

    if n % 2 == 0:
        return (values[n // 2 - 1] + values[n // 2]) // 2
    else:
        return values[n // 2]


def estimate_join_size_with_sketch_plus_minus(a: AMSSketchPlusMinus, b: AMSSketchPlusMinus) -> int:
    # both sketches must have the same number of hash functions
    assert a.array_size == b.array_size

    print(a.array)
    print(b.array)

    sum = 0

    for j in range(a.array_size):
        sum += a.array[j] * b.array[j]

    return sum


def estimate_join_size_with_sketch_counter(a: AMSSketchCounter, b: AMSSketchCounter) -> int:
    # both sketches must have the same number of hash functions
    assert a.n_hash_functions == b.n_hash_functions

    estimates = []

    for i in range(a.n_hash_functions):
        array_a = a.array[i]
        array_b = b.array[i]

        sum = 0

        for j in range(a.array_size):
            sum += array_a[j] * array_b[j]
        estimates.append(sum)

    median_value = median(estimates)
    return max(0, median_value)


def join_relations_and_estimate_size(data: (Relation, Relation)) -> (int, int):
    a, b = data
    joined = join(a, b)
    actual_join_size = len(joined)

    sketch_a = AMSSketchCounter(512, 8)
    sketch_b = AMSSketchCounter(512, 8)

    for value in a:
        hash = hash_int(value)
        sketch_a.update(hash, 1)

    for value in b:
        hash = hash_int(value)
        sketch_b.update(hash, 1)

    estimated_join_size = estimate_join_size_with_sketch_counter(sketch_a, sketch_b)
    return actual_join_size, estimated_join_size


def get_relative_error(actual: int, estimate: int) -> float:
    if actual == 0:
        return 1
    return abs(actual - estimate) / actual


def get_absolute_error(actual: int, estimate: int) -> int:
    return abs(actual - estimate)


def run_experiment_join_size(data_space: List[Relation], n_joins: int) -> pd.DataFrame:
    df = pd.DataFrame()

    join_data = get_join_input(data_space, n_joins)

    df = pd.DataFrame()

    for data in tqdm(join_data):
        a, b = data
        n_a = len(a)
        n_b = len(b)
        n_unique_a = len(set(a))
        n_unique_b = len(set(b))
        actual, estimate = join_relations_and_estimate_size(data)
        relative_error = get_relative_error(actual, estimate)
        absolute_error = get_absolute_error(actual, estimate)

        new_row = pd.DataFrame([{
            'n_a': n_a,
            'n_b': n_b,
            'n_unique_a': n_unique_a,
            'n_unique_b': n_unique_b,
            'actual': actual,
            'estimate': estimate,
            'relative_error': relative_error,
            'absolute_error': absolute_error
        }])

        df = pd.concat([df, new_row], ignore_index=True)

    print(df)

    return df

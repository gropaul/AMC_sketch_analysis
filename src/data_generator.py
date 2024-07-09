import itertools
import random
from math import floor
from typing import List
import xxhash

from src.my_types import Relation


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
    # turn value in bytes
    data = value.to_bytes(8, byteorder='big', signed=False)
    # return a 64-bit hash of the value
    hash_64 = xxhash.xxh64(data).intdigest()
    return hash_64

import math

import numpy as np
import xxhash


def _get_bit_at_int(hash, bit):
    return (hash >> bit) & 1


class AMSSketchPlusMinus:
    def __init__(self, array_size):
        # Initialize the sketch array
        self.array_size = array_size
        self.array = np.zeros(array_size)
        self.updates = 0

    def update(self, hash, w):
        self.updates += 1
        for i in range(self.array_size):
            bit = _get_bit_at_int(hash, i)
            if bit == 0:
                self.array[i] -= w
            else:
                self.array[i] += w


PRIMES = [
    120000000089,
    20000000089,
    30000000091,
    40000000129,
    50000000131,
    60000000139,
    70000000159,
    80000000167,
    90000000191,
    100000000193
]


def get_hash_functions(prime_index: int, value_range: int):
    def hash_function(value):
        hasher = xxhash.xxh64(seed=prime_index)
        data = value.to_bytes(8, byteorder='big', signed=False)
        hasher.update(data)
        hash_value = hasher.intdigest()
        return hash_value % value_range

    return hash_function


def calc_weight(value):
    hasher = xxhash.xxh64(seed=123124)
    data = value.to_bytes(8, byteorder='big', signed=False)
    hasher.update(data)
    hash_value = hasher.intdigest()
    # return either 1 or -1 with equal probability
    return 1 if hash_value % 2 == 0 else -1


class AMSSketchCounter:
    def __init__(self, array_size, n_hash_functions: int = len(PRIMES)):
        # Initialize the sketch array
        self.array_size = array_size
        self.n_hash_functions = n_hash_functions
        self.array = np.zeros((n_hash_functions, array_size))
        self.updates = 0

        self.hash_functions = [get_hash_functions(i, array_size) for i in range(n_hash_functions)]

    def update(self, hash, w):
        self.updates += 1
        for (i, hash_function) in enumerate(self.hash_functions):
            index = hash_function(hash)
            weight = calc_weight(hash)
            self.array[i][index] += weight * w

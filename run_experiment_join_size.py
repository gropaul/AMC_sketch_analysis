import random

from src.data_generator import get_data_space
from src.join_size_sketch import run_experiment_join_size


def main():
    # set the seed
    seed = 42
    random.seed(seed)

    n_rows = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
    n_duplicates = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

    relations = get_data_space(n_rows, n_duplicates)
    random.shuffle(relations)

    df = run_experiment_join_size(relations, n_joins=64)
    # save as csv
    df.to_csv('sketch_results_join.csv', index=False)


if __name__ == '__main__':
    main()

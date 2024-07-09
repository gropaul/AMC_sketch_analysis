import pandas as pd
import matplotlib.pyplot as plt

COLOR_COLUMN = 'n_rows'  # different colors for different n_rows
FEATURE_COLUMN = 'n_duplicates'  # the column to be plotted on the x-axis

def main():
    PATH = '../sketch_results.csv'
    df = pd.read_csv(PATH)

    columns = [column for column in df.columns if column not in [FEATURE_COLUMN]]

    # Determine the number of columns and create a scatter plot for each column, one big figure
    n_columns = len(columns)
    n_rows = 1
    if n_columns > 1:
        fig, axs = plt.subplots(n_rows, n_columns, figsize=(n_columns * 5, n_rows * 5))
    else:
        fig, axs = plt.subplots(n_rows, n_columns, figsize=(5, 5))
        axs = [axs]  # make it iterable for consistency

    for idx, column in enumerate(columns):
        for color, group in df.groupby(COLOR_COLUMN):
            # sort by n_duplicates
            group = group.sort_values(by=FEATURE_COLUMN)
            axs[idx].plot(group[FEATURE_COLUMN], group[column], label=color, marker='o')
        axs[idx].set_title(column)
        axs[idx].set_xlabel(FEATURE_COLUMN)
        axs[idx].set_ylabel(column)
        axs[idx].legend(title=COLOR_COLUMN, loc='upper right')

        # make the x-axis logarithmic
        axs[idx].set_xscale('log')

    plt.tight_layout()  # Adjust layout to make room for plot labels

    plt.savefig('plot.png')
    plt.show()

    # plot the number of rows vs the sketch metrics



if __name__ == '__main__':
    main()

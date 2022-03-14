from TabDataReprGen import main
from multiprocessing import Pool
import sys

# number of files to process overall
num_filenames = 360
modes = ["m"]

filename_indices = list(range(num_filenames))
mode_list = [modes[0]] * 360


if __name__ == "__main__":
    # number of processes will run simultaneously
    pool = Pool(11)
    results = pool.map(main, zip(filename_indices, mode_list))

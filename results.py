import os
import pickle
import texttable as tt
from glob import glob
import numpy as np
import re

from run_neural_logic_mnist import base_result_path

table = tt.Texttable()
table.set_cols_align(["c"] * 10)
table.set_cols_valign(["m"] * 10)
table.set_cols_width(["10"] * 10)
table.set_precision(2)

table.add_row(["split_depth", "num_input_distributions", "num_recursive_splits", "num_sums",
               "dropout_rate_input", "dropout_rate_sums", "num_iterations", 'train_ACC', 'valid_ACC', 'test_ACC'])

files = [y for x in os.walk(base_result_path) for y in glob(os.path.join(x[0], 'results.pkl'))]
# results = np.empty((0, 10))

for file in files:
    parameters = ([float(s) for s in re.findall(r'-?\d+\.?\d*', file)])

    pkl_file = open(file, 'rb')
    res_dict = pickle.load(pkl_file)
    pkl_file.close()

    try:
        pkl_file = open(file.replace('results.pkl', 'results_history.pkl'), 'rb')
        res_history_dict = pickle.load(pkl_file)
        pkl_file.close()
        num_iterations = len(res_history_dict.keys())
    except FileNotFoundError:
        num_iterations = None

    table.add_row([parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], num_iterations, np.max(np.array(res_dict['train_ACC'])),
                   np.max(np.array(res_dict['valid_ACC'])), np.max(np.array(res_dict['test_ACC']))])

    # results = np.append(results, [[0, 0, 0, 0, 0, 0, 0, best_train_ACC, best_valid_ACC, best_test_ACC]], 0)

file = open("results.txt", "w")
file.write(table.draw())
file.close()

print(table.draw() + "\n")

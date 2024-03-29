import os
import sys
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
               "dropout_rate_input", "dropout_rate_sums", "pseudolabels_threshold", "train_ACC", "valid_ACC", "test_ACC"])

files = [y for x in os.walk(base_result_path + sys.argv[1]) for y in glob(os.path.join(x[0], 'results.pkl'))]

for file in files:
    parameters = ([float(s) for s in re.findall(r'-?\d+\.?\d*', file)])

    pkl_file = open(file, 'rb')
    res_dict = pickle.load(pkl_file)
    pkl_file.close()

    table.add_row([parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7],
                   np.max(np.array(res_dict['train_ACC'])), np.max(np.array(res_dict['valid_ACC'])), np.max(np.array(res_dict['test_ACC']))])

file = open("results_" + str(sys.argv[1]) + ".txt", "w")
file.write(table.draw())
file.close()

print(table.draw() + "\n")

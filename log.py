import sys
import pickle
import numpy as np

from run_neural_logic_mnist import base_result_path

config = "split_depth_" + sys.argv[1] + \
         "__num_input_distributions_" + sys.argv[2] + \
         "__num_recursive_splits_" + sys.argv[3] + \
         "__num_sums_" + sys.argv[4] + \
         "__dropout_rate_input_" + sys.argv[5] + \
         "__dropout_rate_sums_" + sys.argv[6] + \
         "/log.pkl"

try:
    pkl_file = open(base_result_path + config, 'rb')
except FileNotFoundError:
    print("Configuration not found!")
    sys.exit(66)

log_dict = pickle.load(pkl_file)
pkl_file.close()

for epoch_id, epoch_dict in enumerate(log_dict):

    x = np.zeros((10, 10))

    for real_elem, pseudolabel in zip(epoch_dict['reals'], epoch_dict['pseudolabels']):
        x[int(real_elem)][int(pseudolabel)] += 1

    print("EPOCH {}, train_ACC: {}".format(epoch_id, round(epoch_dict['train_ACC'], 2)))
    print('----------------------------')
    print(np.round((x.T/x.sum(axis=1)).T, 2))
    print('\n')

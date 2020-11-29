import os
import sys
import pickle
from glob import glob
import numpy as np
import re
import matplotlib.pyplot as plt
import utils

from run_neural_logic_mnist import base_result_path

graphics_dir = 'graphics/num_addends_' + sys.argv[1] + '/'

files = [y for x in os.walk(base_result_path + sys.argv[1]) for y in glob(os.path.join(x[0], 'log.pkl'))]

considered_examples_dict = {}
accuracies_dict = {}

# aggregate results per threshold
for file in files:
    parameters = ([float(s) for s in re.findall(r'-?\d+\.?\d*', file)])
    print('File parameters:', parameters)
    threshold = parameters[-1]

    pkl_file = open(file, 'rb')
    log_dict = pickle.load(pkl_file)
    pkl_file.close()

    considered_examples_temp = np.array([])
    accuracies_temp = np.array([])

    for epoch_dict in log_dict:
        considered_examples_temp = np.append(considered_examples_temp, epoch_dict['considered_examples'])
        accuracies_temp = np.append(accuracies_temp, epoch_dict['valid_ACC'])

    if threshold in considered_examples_dict:
        considered_examples_dict[threshold] = np.add(considered_examples_dict[threshold], considered_examples_temp)
    else:
        considered_examples_dict[threshold] = considered_examples_temp

    if threshold in accuracies_dict:
        accuracies_dict[threshold] = np.add(accuracies_dict[threshold], accuracies_temp)
    else:
        accuracies_dict[threshold] = accuracies_temp

number_of_thresholds = len(files)/len(considered_examples_dict)

# compute means and save graphics
utils.mkdir_p(graphics_dir)

for key, value in considered_examples_dict.items():
    plt.plot(np.true_divide(value, number_of_thresholds), color='b')
    plt.xlabel('epoch')
    plt.ylabel('#(examples)')
    plt.title('Number of considered examples per epoch')
    plt.savefig(graphics_dir + 'examples_' + str(key) + '.png')
    plt.close()

for key, value in accuracies_dict.items():
    plt.plot(np.true_divide(value, number_of_thresholds), color='r')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy on validation set per epoch')
    plt.savefig(graphics_dir + 'accuracy_' + str(key) + '.png')
    plt.close()

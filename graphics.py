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
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
files = [y for x in os.walk(base_result_path + sys.argv[1]) for y in glob(os.path.join(x[0], 'log.pkl'))]

considered_examples_dict = {}
accuracies_dict = {}
labeling_correctness_dict = {}

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
    labeling_correctness_temp = np.array([])

    for epoch_id, epoch_dict in enumerate(log_dict):
        considered_examples_temp = np.append(considered_examples_temp, epoch_dict['considered_examples'])
        accuracies_temp = np.append(accuracies_temp, epoch_dict['valid_ACC'])

        x = np.zeros((10, 10))
        for real_elem, pseudolabel in zip(epoch_dict['labels'], epoch_dict['pseudolabels']):
            x[int(real_elem)][int(pseudolabel)] += 1

        diagonal = (np.round((x.T / x.sum(axis=1)).T, 2)).diagonal()
        labeling_correctness_temp = np.append(labeling_correctness_temp, np.mean(diagonal[~np.isnan(diagonal)]))

    if threshold in considered_examples_dict:
        considered_examples_dict[threshold] = np.add(considered_examples_dict[threshold], considered_examples_temp)
    else:
        considered_examples_dict[threshold] = considered_examples_temp

    if threshold in accuracies_dict:
        accuracies_dict[threshold] = np.add(accuracies_dict[threshold], accuracies_temp)
    else:
        accuracies_dict[threshold] = accuracies_temp

    if threshold in labeling_correctness_dict:
        labeling_correctness_dict[threshold] = np.add(labeling_correctness_dict[threshold], labeling_correctness_temp)
    else:
        labeling_correctness_dict[threshold] = labeling_correctness_temp

number_of_thresholds = len(files) / len(considered_examples_dict)

# compute means and save graphics
utils.mkdir_p(graphics_dir)

for idx, (key, value) in enumerate(considered_examples_dict.items()):
    plt.plot(np.true_divide(value, number_of_thresholds), color=colors[idx], label=str(key))

plt.legend(loc="lower right")
plt.xlabel('epoch')
plt.ylabel('#(examples)')
plt.title('Number of considered examples per epoch')
plt.savefig(graphics_dir + 'examples.png', dpi=400)
plt.close()

for idx, (key, value) in enumerate(accuracies_dict.items()):
    plt.plot(np.true_divide(value, number_of_thresholds), color=colors[idx], label=str(key))

plt.legend(loc="lower right")
plt.xlabel('epoch')
plt.ylabel('accuracy %')
plt.title('Accuracy on validation set in percentage per epoch')
plt.savefig(graphics_dir + 'accuracy.png', dpi=400)
plt.close()

for idx, (key, value) in enumerate(labeling_correctness_dict.items()):
    plt.plot((np.true_divide(value, number_of_thresholds)) * 100, color=colors[idx], label=str(key))

plt.legend(loc="lower right")
plt.xlabel('epoch')
plt.ylabel('labeling accuracy %')
plt.title('Labeling accuracy in percentage per epoch')
plt.savefig(graphics_dir + 'labeling.png', dpi=400)
plt.close()

import utils
import os
import sys
import filelock
import subprocess
import time

from run_rat_spn_mnist import structure_dict, param_configs, start_time, time_limit_seconds, dont_start_if_less_than_seconds, num_epochs

base_result_path = "results/neural/mnist/num_addends_"
num_addends = 3
pseudolabels_threshold = 0.01


def run():
    total_dict_len = 0
    for _, value in structure_dict.items():
        total_dict_len += len(value)
    total_config = total_dict_len * len(param_configs)
    current_config = 1

    for split_depth in structure_dict:
        for structure_config in structure_dict[split_depth]:
            for config_dict in param_configs:

                remaining_time = time_limit_seconds - (time.time() - start_time)
                if remaining_time < dont_start_if_less_than_seconds:
                    print("Only {} seconds remaining, stop worker".format(remaining_time))
                    sys.exit(0)

                cmd = "python -W ignore train_neural_logic_rat_spn.py --store_best_valid_loss --store_best_valid_acc --num_epochs {}".format(num_epochs)
                cmd += " --timeout_seconds {}".format(remaining_time)
                cmd += " --split_depth {}".format(split_depth)
                cmd += " --data_path data/mnist/"
                cmd += " --num_addends {}".format(num_addends)
                cmd += " --pseudolabels_threshold {}".format(pseudolabels_threshold)

                for key in sorted(structure_config.keys()):
                    cmd += " --{} {}".format(key, structure_config[key])
                for key in sorted(config_dict.keys()):
                    cmd += " --{} {}".format(key, config_dict[key])

                comb_string = ""
                comb_string += "split_depth_{}".format(split_depth)
                for key in sorted(structure_config.keys()):
                    comb_string += "__{}_{}".format(key, structure_config[key])
                for key in sorted(config_dict.keys()):
                    comb_string += "__{}_{}".format(key, config_dict[key])
                comb_string += "__pseudolabels_threshold_{}".format(pseudolabels_threshold)

                result_path = base_result_path + str(num_addends) + "/" + comb_string
                cmd += " --result_path {}".format(result_path)

                ###
                print("Configuration: {}/{}".format(current_config, total_config))
                print(cmd)

                utils.mkdir_p(result_path)
                lock_file = result_path + "/file.lock"
                done_file = result_path + "/file.done"
                lock = filelock.FileLock(lock_file)
                try:
                    lock.acquire(timeout=0.1)
                    if os.path.isfile(done_file):
                        print("   already done -> skip")
                    else:
                        sys.stdout.flush()
                        ret_val = subprocess.call(cmd, shell=True)
                        if ret_val == 7:
                            lock.release()
                            print("Task timed out, stop worker")
                            sys.exit(0)
                        os.system("touch {}".format(done_file))
                    lock.release()
                except filelock.Timeout:
                    print("   locked -> skip")

                current_config += 1


if __name__ == '__main__':
    run()

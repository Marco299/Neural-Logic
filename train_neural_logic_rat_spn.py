import utils
import pickle
import datasets
import numpy as np
import tensorflow as tf
import time
from models.RegionGraph import RegionGraph
from models.RatSpn import RatSpn
from train_rat_spn import make_parser, compute_performance
from pyswip import *

abductions_list_dir = 'abductions/'


def compute_prediction(sess, data_x, batch_size, spn):

    num_batches = int(np.ceil(float(data_x.shape[0]) / float(batch_size)))
    test_idx = 0
    # ma modifica 10
    out_total = np.empty((0, 10))
    pred_total = np.array([])

    for test_k in range(0, num_batches):
        if test_k + 1 < num_batches:
            batch_data = data_x[test_idx:test_idx + batch_size, :]
        else:
            batch_data = data_x[test_idx:, :]

        feed_dict = {spn.inputs: batch_data}
        if spn.dropout_input_placeholder is not None:
            feed_dict[spn.dropout_input_placeholder] = 1.0
        if spn.dropout_sums_placeholder is not None:
            feed_dict[spn.dropout_sums_placeholder] = 1.0

        out_tmp, pred = sess.run(
            [spn.outputs,
             spn.prediction],
            feed_dict=feed_dict)

        out_total = np.append(out_total, out_tmp, 0)
        pred_total = np.append(pred_total, pred, 0)

        test_idx += batch_size

    return out_total, pred_total


def select_pseudolabels(abductions, outputs, train_labels):
    pseudolabels = np.array([])
    examples_indexes = np.array([])

    for idx, digit in enumerate(abductions):
        best_prob = None
        second_best_prob = None
        start_idx = ARGS.num_addends * idx

        for abduction in digit:

            # compute abduction probability
            abduction_prob = outputs[start_idx][abduction[0]]
            for i in range(1, ARGS.num_addends):
                abduction_prob += outputs[start_idx+i][abduction[i]]

            # update the two most likely abductions
            if best_prob is None:
                best_prob = abduction_prob
                best_abduction = [abduction[i] for i in range(0, ARGS.num_addends)]
            elif second_best_prob is None:
                if abduction_prob <= best_prob:
                    second_best_prob = abduction_prob
                else:
                    second_best_prob = best_prob
                    best_prob = abduction_prob
                    best_abduction = [abduction[i] for i in range(0, ARGS.num_addends)]
            else:
                if abduction_prob > best_prob:
                    second_best_prob = best_prob
                    best_prob = abduction_prob
                    best_abduction = [abduction[i] for i in range(0, ARGS.num_addends)]
                elif abduction_prob > second_best_prob:
                    second_best_prob = abduction_prob

        # compute variation between the two most likely abductions
        if second_best_prob is not None:
            variation = abs((best_prob - second_best_prob) / best_prob * 100)

        # filter results according to the threshold
        if second_best_prob is None or variation >= ARGS.pseudolabels_threshold:
            pseudolabels = np.append(pseudolabels, best_abduction, 0)
            examples_indexes = np.append(examples_indexes, [start_idx+i for i in range(0, ARGS.num_addends)], 0)

    return pseudolabels, examples_indexes.astype(int)


def run_training():
    training_start_time = time.time()

    #############
    # Load data #
    #############
    if ARGS.data_set in ['mnist']:
        train_x, train_labels, valid_x, valid_labels, test_x, test_labels = datasets.load_mnist(ARGS.data_path)
    else:
        print("Cannot find dataset")
        sys.exit(0)

    ######################
    # Data preprocessing #
    ######################
    if not ARGS.discrete_leaves:
        if ARGS.low_variance_threshold >= 0.0:
            v = np.var(train_x, 0)
            mu = np.mean(v)
            idx = v > ARGS.low_variance_threshold * mu
            train_x = train_x[:, idx]
            test_x = test_x[:, idx]
            if valid_x is not None:
                valid_x = valid_x[:, idx]

        # zero-mean, unit-variance
        if ARGS.normalization == "zmuv":
            train_x_mean = np.mean(train_x, 0)
            train_x_std = np.std(train_x, 0)

            train_x = (train_x - train_x_mean) / (train_x_std + ARGS.zmuv_min_sigma)
            test_x = (test_x - train_x_mean) / (train_x_std + ARGS.zmuv_min_sigma)
            if valid_x is not None:
                valid_x = (valid_x - train_x_mean) / (train_x_std + ARGS.zmuv_min_sigma)

    num_classes = len(np.unique(train_labels))
    train_n = int(train_x.shape[0])
    num_dims = int(train_x.shape[1])

    # stores evaluation metrics
    results = {
        'train_ACC': [],
        'train_CE': [],
        'train_LL': [],
        'train_MARG': [],
        'test_ACC': [],
        'test_CE': [],
        'test_LL': [],
        'test_MARG': [],
        'valid_ACC': [],
        'valid_CE': [],
        'valid_LL': [],
        'valid_MARG': [],
        'elapsed_wall_time_epoch': [],
        'best_valid_acc': None,
        'epoch_best_valid_acc': None,
        'best_valid_loss': None,
        'epoch_best_valid_loss': None
    }

    # stores log info
    logs = []

    # Make Region Graph
    region_graph = RegionGraph(range(0, num_dims), np.random.randint(0, 1000000000))
    for _ in range(0, ARGS.num_recursive_splits):
        region_graph.random_split(2, ARGS.split_depth)
    region_graph_layers = region_graph.make_layers()

    # Make Tensorflow model
    rat_spn = RatSpn(region_graph_layers, num_classes, ARGS=ARGS)

    if not ARGS.no_save:
        pickle.dump((num_dims, num_classes, ARGS, region_graph_layers),
                    open(ARGS.result_path + '/spn_description.pkl', "wb"))

    # session
    if ARGS.GPU_fraction <= 0.95:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=ARGS.GPU_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = tf.Session()

    # saver
    saver = tf.train.Saver(max_to_keep=ARGS.store_model_max)
    if ARGS.store_best_valid_acc:
        best_valid_acc_saver = tf.train.Saver(max_to_keep=1)
    if ARGS.store_best_valid_loss:
        best_valid_loss_saver = tf.train.Saver(max_to_keep=1)

    # init model
    init = tf.global_variables_initializer()
    sess.run(init)
    if ARGS.model_init_file:
        init_saver = tf.train.Saver(rat_spn.all_params)
        init_saver.restore(sess, ARGS.model_init_file)
        print("")
        print("used {} to init model".format(ARGS.model_init_file))
        print("")

    # print(rat_spn)

    abductions_list_file = abductions_list_dir + 'abductions_list_' + str(ARGS.num_addends) + '.pkl'

    if os.path.exists(abductions_list_file):
        print("Load abductions")
        abductions_list = pickle.load(open(abductions_list_file, 'rb'))
    else:
        print("Compute and save abductions")
        train_sums = [int(sum(train_labels[current: current + ARGS.num_addends])) for current in range(0, len(train_labels), ARGS.num_addends)]
        prolog = Prolog()
        prolog.consult("abduction.pl")
        generate_abductions = Functor("generate_abductions", 3)
        x = Variable()
        q = Query(generate_abductions(train_sums, ARGS.num_addends, x))
        q.nextSolution()
        abductions_list = x.value
        q.closeQuery()
        pickle.dump(abductions_list, open(abductions_list_file, "wb"))

    train_sums_n = len(abductions_list)
    """
    single element(couple): abductions_list[i]
    indexes: abductions_list[i][0]
    abductions: abductions_list[i][1]
    """

    ############
    # Training #
    ############

    print("Start training")

    all_labels = all_pseudolabels = np.array([])
    # all_outputs = np.empty((0, 10))

    epoch_elapsed_times = []

    batch_size = ARGS.batch_size
    while batch_size % ARGS.num_addends != 0:
        batch_size += 1

    for epoch_n in range(0, ARGS.num_epochs):

        epoch_start_time = time.time()

        batch_idx = 0
        elapsed_wall_time_epoch = 0.0
        considered_examples = 0

        while batch_idx != train_sums_n:
            cur_idx = []
            cur_abductions = []
            while True:
                cur_idx += abductions_list[batch_idx][0]
                cur_abductions.append(abductions_list[batch_idx][1])

                batch_idx += 1
                if batch_idx == train_sums_n or \
                        len(abductions_list[batch_idx-1][1]) != len(abductions_list[batch_idx][1])\
                        or len(cur_idx) == batch_size:
                    break

            outputs, prediction = compute_prediction(
                sess,
                train_x[cur_idx, :],
                len(cur_idx),
                rat_spn)

            pseudolabels, indexes = select_pseudolabels(cur_abductions, outputs, train_labels[cur_idx])
            cur_idx = np.array(cur_idx)[indexes].tolist()

            all_labels = np.append(all_labels, train_labels[cur_idx], 0)
            all_pseudolabels = np.append(all_pseudolabels, pseudolabels, 0)
            considered_examples += len(cur_idx)

            feed_dict = {rat_spn.inputs: train_x[cur_idx, :], rat_spn.labels: pseudolabels}

            if ARGS.dropout_rate_input is not None:
                feed_dict[rat_spn.dropout_input_placeholder] = ARGS.dropout_rate_input
            if ARGS.dropout_rate_sums is not None:
                feed_dict[rat_spn.dropout_sums_placeholder] = ARGS.dropout_rate_sums

            start_time = time.time()
            if ARGS.optimizer == "em":
                one_hot_labels = -np.inf * np.ones((len(cur_idx), num_classes))
                one_hot_labels[range(len(cur_idx)), [int(x) for x in pseudolabels]] = 0.0
                feed_dict[rat_spn.EM_deriv_input_pl] = one_hot_labels

                start_time = time.time()
                sess.run(rat_spn.em_update_accums, feed_dict=feed_dict)
                elapsed_wall_time_epoch += (time.time() - start_time)
            else:
                _, CEM_value, cur_lr, loss_val, ll_mean_val, margin_val = \
                    sess.run([
                        rat_spn.train_op,
                        rat_spn.cross_entropy_mean,
                        rat_spn.learning_rate,
                        rat_spn.objective,
                        rat_spn.neg_norm_ll,
                        rat_spn.neg_margin_objective], feed_dict=feed_dict)
                elapsed_wall_time_epoch += (time.time() - start_time)

        if ARGS.optimizer == "em":
            sess.run(rat_spn.em_update_params)
            sess.run(rat_spn.em_reset_accums)
        else:
            sess.run(rat_spn.decrease_lr_op)

        ################
        ### Evaluate ###
        ################
        print('')
        print('epoch {}'.format(epoch_n))
        print('considered examples: {}'.format(considered_examples))

        num_correct_train, CE_total, train_LL, train_MARG, train_loss = compute_performance(
            sess,
            train_x,
            train_labels,
            100,
            rat_spn)
        train_ACC = 100. * float(num_correct_train) / float(train_x.shape[0])
        train_CE = CE_total / float(train_x.shape[0])
        print('   ###')
        print('   ### accuracy on train set = {}   CE = {}   LL: {}   negmargin: {}'.format(
            train_ACC,
            train_CE,
            train_LL,
            train_MARG))

        if test_x is not None:
            num_correct_test, CE_total, test_LL, test_MARG, test_loss = compute_performance(
                sess,
                test_x,
                test_labels,
                100,
                rat_spn)
            test_ACC = 100. * float(num_correct_test) / float(test_x.shape[0])
            test_CE = CE_total / float(test_x.shape[0])
            print('   ###')
            print(
                '   ### accuracy on test set = {}   CE = {}   LL: {}   negmargin: {}'.format(test_ACC, test_CE, test_LL,
                                                                                             test_MARG))
        else:
            test_ACC = None
            test_CE = None
            test_LL = None

        if valid_x is not None:
            num_correct_valid, CE_total, valid_LL, valid_MARG, valid_loss = compute_performance(
                sess,
                valid_x,
                valid_labels,
                100,
                rat_spn)
            valid_ACC = 100. * float(num_correct_valid) / float(valid_x.shape[0])
            valid_CE = CE_total / float(valid_x.shape[0])
            print('   ###')
            print('   ### accuracy on valid set = {}   CE = {}   LL: {}   margin: {}'.format(
                valid_ACC,
                valid_CE,
                valid_LL,
                valid_MARG))
        else:
            valid_ACC = None
            valid_CE = None
            valid_LL = None

        print('   ###')
        print('')

        # timing
        epoch_elapsed_times.append(time.time() - epoch_start_time)

        #logs
        logs.append({'valid_ACC': valid_ACC,
                     'labels': all_labels,
                     'pseudolabels': all_pseudolabels,
                     'considered_examples': considered_examples
                     })

        if not ARGS.no_save:
            results['train_ACC'].append(train_ACC)
            results['train_CE'].append(train_CE)
            results['train_LL'].append(train_LL)
            results['train_MARG'].append(train_LL)
            results['test_ACC'].append(test_ACC)
            results['test_CE'].append(test_CE)
            results['test_LL'].append(test_LL)
            results['test_MARG'].append(train_LL)
            results['valid_ACC'].append(valid_ACC)
            results['valid_CE'].append(valid_CE)
            results['valid_LL'].append(valid_LL)
            results['valid_MARG'].append(train_LL)
            results['elapsed_wall_time_epoch'].append(elapsed_wall_time_epoch)

            if ARGS.store_best_valid_acc and valid_x is not None:
                if results['best_valid_acc'] is None or valid_ACC > results['best_valid_acc']:
                    print('Better validation accuracy -> save model')
                    print('')

                    best_valid_acc_saver.save(
                        sess,
                        ARGS.result_path + "/best_valid_acc/model.ckpt",
                        global_step=epoch_n,
                        write_meta_graph=False)

                    results['best_valid_acc'] = valid_ACC
                    results['epoch_best_valid_acc'] = epoch_n

            if ARGS.store_best_valid_loss and valid_x is not None:
                if results['best_valid_loss'] is None or valid_loss < results['best_valid_loss']:
                    print('Better validation loss -> save model')
                    print('')

                    best_valid_loss_saver.save(
                        sess,
                        ARGS.result_path + "/best_valid_loss/model.ckpt",
                        global_step=epoch_n,
                        write_meta_graph=False)

                    results['best_valid_loss'] = valid_loss
                    results['epoch_best_valid_loss'] = epoch_n

            if epoch_n % ARGS.store_model_every_epochs == 0 \
                    or epoch_n + 1 == ARGS.num_epochs:
                pickle.dump(results, open(ARGS.result_path + '/results.pkl', "wb"))
                saver.save(sess, ARGS.result_path + "/checkpoints/model.ckpt", global_step=epoch_n,
                           write_meta_graph=False)
                pickle.dump(logs, open(ARGS.result_path + '/log.pkl', "wb"))


if __name__ == '__main__':
    parser = make_parser()

    ARGS = parser.parse_args()

    #
    if not ARGS.no_save:
        utils.mkdir_p(ARGS.result_path)

    # set learning rate
    if ARGS.provided_learning_rate is None:
        if ARGS.optimizer == "adam":
            ARGS.provided_learning_rate = 0.001
        elif ARGS.optimizer == "momentum":
            ARGS.provided_learning_rate = 2.0
        elif ARGS.optimizer == "em":
            pass
        else:
            raise NotImplementedError("Unknown optimizer.")

    # process dropout_rate params
    if ARGS.dropout_rate_input is not None:
        if ARGS.dropout_rate_input >= 1.0 or ARGS.dropout_rate_input <= 0.0:
            ARGS.dropout_rate_input = None

        # process dropout_rate params
    if ARGS.dropout_rate_sums is not None:
        if ARGS.dropout_rate_sums >= 1.0 or ARGS.dropout_rate_sums <= 0.0:
            ARGS.dropout_rate_sums = None

    # process lambda_discriminative and kappa_discriminative params
    ARGS.lambda_discriminative = min(max(ARGS.lambda_discriminative, 0.0), 1.0)
    ARGS.kappa_discriminative = min(max(ARGS.kappa_discriminative, 0.0), 1.0)

    # print ARGS
    sorted_keys = sorted(ARGS.__dict__.keys())
    for k in sorted_keys:
        print('{}: {}'.format(k, ARGS.__dict__[k]))
    print("")

    utils.mkdir_p(abductions_list_dir)

    run_training()

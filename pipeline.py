import os

from river.anomaly import HalfSpaceTrees
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from pipeline_vfhst import Pipeline
from stream_framework import run_stream
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, precision_score, f1_score, average_precision_score
from river import anomaly, compose, preprocessing
from vfhst import VeryFastHalfSpaceTrees
import time
import torch

import pandas as pd


def pipeline_test(
    n_trees,
    height,
    window_size,
    thresholds,
    weights,
    seeds,
    window_already_seen,
    params,
    limits,
    output_file,
    val_times,
    test_times,
    val_and_test,
    g,
    model,
    device,
    emb):
    roc_auc_list = []
    ap_results_list = []
    balanced_accuracy_list = []
    f1_list = []
    precision_list = []
    recall_tn_list = []
    recall_tp_list = []
    timings = []

    scores_and_labels_df = []
    roc_auc_over_time_df = []

    evaluate_cyber_attacks = False

    if params.dataset == "DARPA":

        cyber_attacks_results = {'dos': [[], []], 'probe': [[], []],
                                  'r2l': [[], []], 'u2r': [[], []]}

    elif params.dataset == "UNSW-NB15":

        cyber_attacks_results = {'Exploits': [[], []], 'Reconnaissance': [[], []],
                                  'DoS': [[], []], 'Shellcode': [[], []],
                                  'Fuzzers': [[], []], 'Worms': [[], []],
                                  'Backdoors': [[], []]}

    for i, seed in enumerate(seeds):
        params.seed = seed
        np.random.seed(seed)

        # ROBUSTNESS EXPERIMENT
        # size = int(len(test_times) * 0.5)
        # idx_sampled = np.sort(np.random.choice(np.arange(len(test_times)), size=size, replace=False))
        # times = np.concatenate((val_times, test_times[idx_sampled]))

        times = np.concatenate((val_times, test_times))

        threshold = thresholds[i]

        g_tmp = g.clone()

        hst_edge = VeryFastHalfSpaceTrees(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            seed=params.seed,
            limits=limits
        )

        hst_node = VeryFastHalfSpaceTrees(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            seed=params.seed,
            limits=limits
        )

        hst_edge_model = Pipeline(
            preprocessing.MinMaxScaler(),
            hst_edge
        )

        hst_node_model = Pipeline(
            preprocessing.MinMaxScaler(),
            hst_node
        )

        if weights[1] > 0.0:
            for idx, x in enumerate(emb):
                features = {}
                for i in range(params.out_channels):
                    features[i] = x[i]

                hst_node_model['MinMaxScaler'].learn_one(
                    features)

            if window_size >= emb.shape[0]:
                emb_tmp = emb
            elif params.random_start:
                np.random.seed(seed)
                emb_indexes = np.random.choice(
                    np.arange(emb.shape[0]), (window_size + window_size // 2), replace=False)
                emb_tmp = emb[emb_indexes, :]
            else:
                emb_tmp = emb[-(window_size + window_size // 2):, :]

            for idx, x in enumerate(emb_tmp):
                features = {}
                for i in range(params.out_channels):
                    features[i] = x[i]

                if emb_tmp.shape[0] < window_size and idx == emb_tmp.shape[0] - 1:
                    hst_node.counter = window_size - 1

                hst_node_model.learn_one(features)

        edges_unique = g_tmp.edge_index.t().cpu()

        if params.phi == "mean":
            edge_emb = (emb[edges_unique[:, 0]] + emb[edges_unique[:, 1]]) / 2
        else:
            edge_emb = (emb[edges_unique[:, 1]] - emb[edges_unique[:, 0]])

        if type(edge_emb) != np.ndarray:
            edge_emb = edge_emb.numpy()

        for idx, x in enumerate(edge_emb):
            features = {}
            for i in range(params.out_channels):
                features[i] = x[i]
            hst_edge_model['MinMaxScaler'].learn_one(features)

        if window_size >= edge_emb.shape[0]:
            edge_emb_tmp = emb
        elif params.random_start:
            np.random.seed(seed)

            edges_indexes = np.random.choice(np.arange(edge_emb.shape[0]), (window_size + window_size // 2),
                                             replace=False)
            edge_emb_tmp = edge_emb[edges_indexes, :]
        else:
            edge_emb_tmp = edge_emb[-(window_size + window_size // 2):, :]

        for idx, x in enumerate(edge_emb_tmp):
            features = {}
            for i in range(params.out_channels):
                features[i] = x[i]

            if edge_emb_tmp.shape[0] < window_size and idx == edge_emb_tmp.shape[0] - 1:
                hst_edge.counter = window_size - 1

            hst_edge_model.learn_one(features)

        start_test_index = -1
        if "CL" in params.model_save_path:
            if params.dataset == "DARPA":
                start_test_index = 2275809
            elif params.dataset == "UNSW-NB15":
                start_test_index = 1270627
            elif params.dataset == "ISCX":
                start_test_index = 548095
            elif params.dataset == "IDS2018":
                start_test_index = 3976526
            elif params.dataset == "CTU-13-Scenario-1":
                start_test_index = 1412124
            elif params.dataset == "CTU-13-Scenario-10":
                start_test_index = 654811
            elif params.dataset == "CTU-13-Scenario-13":
                start_test_index = 963320
        else:
            if params.dataset == "DARPA":
                start_test_index = 3645696
            elif params.dataset == "UNSW-NB15":
                start_test_index = 2032074
            elif params.dataset == "ISCX":
                start_test_index = 877662
            elif params.dataset == "IDS2018":
                start_test_index = 6359183
            elif params.dataset == "CTU-13-Scenario-1":
                start_test_index = 2259709
            elif params.dataset == "CTU-13-Scenario-10":
                start_test_index = 1047832
            elif params.dataset == "CTU-13-Scenario-13":
                start_test_index = 1540118

        tp, tn, fp, fn, scores_and_labels, timing, roc_auc_over_time, counts_for_each_attack = run_stream(params, times, val_and_test, g_tmp, model,
                                                                      weights, hst_edge_model, hst_node_model, device,
                                                                      window_already_seen, start_test_index=start_test_index,
                                                                      threshold=threshold, evaluate_cyber_attacks=evaluate_cyber_attacks)

        scores_and_labels = np.array(scores_and_labels)
        roc_auc_results = roc_auc_score(scores_and_labels[:, 1], scores_and_labels[:, 0])
        roc_auc_results = np.around(roc_auc_results, 15)

        for i in range(len(roc_auc_over_time)):
            roc_auc_over_time_df.append((roc_auc_over_time[i], seed))

        ap_results = average_precision_score(scores_and_labels[:, 1], scores_and_labels[:, 0])
        ap_results = np.around(ap_results, 15)

        for i in range(len(scores_and_labels)):
            scores_and_labels_df.append((scores_and_labels[i, 1], scores_and_labels[i, 0], seed))

        precision = np.around((tp) / (tp + fp), 15) if (tp + fp) > 0 else 0
        recall_tp = np.around((tp) / (tp + fn), 15) if (tp + fn) > 0 else 0
        recall_tn = np.around((tn) / (tn + fp), 15) if (tn + fp) > 0 else 0
        balanced_accuracy = np.around((recall_tp + recall_tn) / 2, 15)
        f1score = np.around((2 * precision * recall_tp) / (precision + recall_tp), 15) \
            if (precision + recall_tp) > 0 else 0

        roc_auc_list.append(roc_auc_results)
        ap_results_list.append(ap_results)
        balanced_accuracy_list.append(balanced_accuracy)
        f1_list.append(f1score)
        precision_list.append(precision)
        recall_tp_list.append(recall_tp)
        recall_tn_list.append(recall_tn)
        timings.append(timing)

        if params.dataset in ["DARPA", "UNSW-NB15"] and evaluate_cyber_attacks:
            # Cyber attacks types
            for k in counts_for_each_attack.keys():
                tp = counts_for_each_attack[k][0]
                tn = counts_for_each_attack[k][1]
                fp = counts_for_each_attack[k][2]
                fn = counts_for_each_attack[k][3]

                #print(k)

                precision = np.around((tp) / (tp + fp), 15) if (tp + fp) > 0 else 0
                recall_tp = np.around((tp) / (tp + fn), 15) if (tp + fn) > 0 else 0
                recall_tn = np.around((tn) / (tn + fp), 15) if (tn + fp) > 0 else 0
                balanced_accuracy = np.around((recall_tp + recall_tn) / 2, 15)
                f1score = np.around((2 * precision * recall_tp) / (precision + recall_tp), 15) \
                    if (precision + recall_tp) > 0 else 0

                if len(np.unique(counts_for_each_attack[k][4])) == 2:
                    auc = roc_auc_score(counts_for_each_attack[k][4], counts_for_each_attack[k][5])
                else:
                    auc = 1.0
                ap = average_precision_score(counts_for_each_attack[k][4], counts_for_each_attack[k][5])

                cyber_attacks_results[k][0].append(auc)
                cyber_attacks_results[k][1].append(ap)

        print("TEST")
        print("Model", params.model_name, "seed", params.seed, "n_trees", n_trees, "\theight", height, "\twindow_size",
            window_size, "\tthreshold", threshold, "Weights scores", weights, "Window Already Seen", window_already_seen,
            "ROC-AUC: {}".format(roc_auc_results), "AP: {}".format(ap_results), "Balanced Accuracy", balanced_accuracy, "F1", f1score)
        print("True Positive", tp, " True Negative", tn, " False Positive", fp, " False Negative", fn)


        if not os.path.exists(output_file):
            f = open(output_file, "w")
            f.write(
                "Seed\tn_trees\theight\twindow_size\tthreshold\tWeights\tROC-AUC\tAP\tF1-Score\tBalancedAccuracy\tRecall\tPrecision\n")
            f.close()

        f = open(output_file, "a")
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(params.seed, n_trees, height,
                                                                  window_size, threshold, weights,
                                                                  roc_auc_results, ap_results, f1score, balanced_accuracy,
                                                                  recall_tp, precision))
        f.close()
        



    print("Model", params.model_name, "n_trees", n_trees, "\theight", height, "\twindow_size", window_size,
            "\tthreshold", thresholds, "Weights scores", weights, "Window Already Seen", window_already_seen)

    print("ROC: {} +- {}".format(np.mean(roc_auc_list), np.std(roc_auc_list)))
    print("AP: {} +- {}".format(np.mean(ap_results_list), np.std(ap_results_list)))
    print("F1: {} +- {}".format(np.mean(f1_list), np.std(f1_list)))
    print("Balanced Acc: {} +- {}".format(np.mean(balanced_accuracy_list),
          np.std(balanced_accuracy_list)))
    print("Precision: {} +- {}".format(np.mean(precision_list), np.std(precision_list)))
    print("Recall: {} +- {}".format(np.mean(recall_tp_list), np.std(recall_tp_list)))
    print("Timing Mean: {}".format(np.mean(timings)))

    if params.dataset in ["DARPA", "UNSW-NB15"] and evaluate_cyber_attacks:
        for k in counts_for_each_attack.keys():
            print(k)
            print("ROC: {} +- {}".format(np.mean(cyber_attacks_results[k][0]), np.std(cyber_attacks_results[k][0])))
            print("AP: {} +- {}".format(np.mean(cyber_attacks_results[k][1]), np.std(cyber_attacks_results[k][1])))


def pipeline_validation(
    n_trees_list,
    height_list,
    window_size_list,
    weights_scores,
    seeds,
    windows_already_seen,
    params,
    limits,
    times,
    val,
    g,
    model,
    device,
     emb):

    min_roc_value_early_stopping = 0
    if params.dataset == "DARPA":
        min_roc_value_early_stopping = 0.9#0.96
    if params.dataset == "UNSW-NB15":
        min_roc_value_early_stopping = 0.9#0.84
    elif params.dataset == "ISCX":
        min_roc_value_early_stopping = 0.9
    elif params.dataset == "IDS2018":
        min_roc_value_early_stopping = 0.97
    elif params.dataset == "CTU-13-Scenario-1":
        min_roc_value_early_stopping = 0.9#0.89
    elif params.dataset == "CTU-13-Scenario-10":
        min_roc_value_early_stopping = 0.86
    elif params.dataset == "CTU-13-Scenario-13":
        min_roc_value_early_stopping = 0.75#0.86

    for weight_scores in weights_scores:
        for window_already_seen in windows_already_seen:
            for n_trees in n_trees_list:
                for height in height_list:
                    for window_size in window_size_list:

                        roc_auc_list = []
                        timings = []
                        all_scores_and_labels = []
                        ap_results_list = []

                        roc_auc_results = 0

                        for seed in seeds:
                            params.seed = seed
                            np.random.seed(seed)

                            g_tmp = g.clone().to(device)

                            hst_edge = VeryFastHalfSpaceTrees(
                                n_trees=n_trees,
                                height=height,
                                window_size=window_size,
                                seed=params.seed,
                                limits=limits
                            )

                            hst_node = VeryFastHalfSpaceTrees(
                                n_trees=n_trees,
                                height=height,
                                window_size=window_size,
                                seed=params.seed,
                                limits=limits
                            )

                            hst_edge_model = Pipeline(
                                preprocessing.MinMaxScaler(),
                                hst_edge
                            )

                            hst_node_model = Pipeline(
                                preprocessing.MinMaxScaler(),
                                hst_node
                            )

                            if weight_scores[1] > 0.0:
                                for idx, x in enumerate(emb):
                                    features = {}
                                    for i in range(params.out_channels):
                                        features[i] = x[i]

                                    hst_node_model['MinMaxScaler'].learn_one(
                                        features)

                                if window_size >= emb.shape[0]:
                                    emb_tmp = emb
                                elif params.random_start:
                                    np.random.seed(seed)
                                    emb_indexes = np.random.choice(
                                        np.arange(emb.shape[0]), (window_size + window_size // 2), replace=False)
                                    emb_tmp = emb[emb_indexes, :]
                                else:
                                    emb_tmp = emb[-(window_size + window_size //2):, :]

                                for idx, x in enumerate(emb_tmp):
                                    features = {}
                                    for i in range(params.out_channels):
                                        features[i] = x[i]

                                    if emb_tmp.shape[0] < window_size and idx == emb_tmp.shape[0] - 1:
                                        hst_node.counter = window_size - 1

                                    hst_node_model.learn_one(features)

                            edges_unique = g_tmp.edge_index.t().cpu()

                            if params.phi == "mean":
                                edge_emb = (emb[edges_unique[:, 0]] + emb[edges_unique[:, 1]]) / 2
                            else:
                                edge_emb = (emb[edges_unique[:, 1]] - emb[edges_unique[:, 0]])

                            if type(edge_emb) != np.ndarray:
                                edge_emb = edge_emb.numpy()

                            for idx, x in enumerate(edge_emb):
                                features = {}
                                for i in range(params.out_channels):
                                    features[i] = x[i]
                                hst_edge_model['MinMaxScaler'].learn_one(features)

                            if window_size >= edge_emb.shape[0]:
                                edge_emb_tmp = edge_emb
                            elif params.random_start:
                                np.random.seed(seed)
                                edges_indexes = np.random.choice(np.arange(edge_emb.shape[0]), (window_size + window_size //2), replace=False)
                                edge_emb_tmp = edge_emb[edges_indexes, :]
                            else:
                                edge_emb_tmp = edge_emb[-(window_size + window_size //2):, :]

                            for idx, x in enumerate(edge_emb_tmp):
                                features = {}
                                for i in range(params.out_channels):
                                    features[i] = x[i]

                                if edge_emb_tmp.shape[0] < window_size and idx == edge_emb_tmp.shape[0] - 1:
                                    hst_edge.counter = window_size - 1

                                hst_edge_model.learn_one(features)

                            _, _, _, _, scores_and_labels, timing, _, _ = run_stream(params, times, val, g_tmp, model,
                                                                               weight_scores, hst_edge_model,
                                                                               hst_node_model, device, window_already_seen)

                            timings.append(timing)

                            all_scores_and_labels.append(scores_and_labels)

                            scores_and_labels = np.array(scores_and_labels)

                            roc_auc_results = roc_auc_score(
                                scores_and_labels[:, 1], scores_and_labels[:, 0])
                            roc_auc = np.around(roc_auc_results, 3)

                            roc_auc_list.append(roc_auc)

                            ap_results = average_precision_score(scores_and_labels[:, 1], scores_and_labels[:, 0])
                            ap_results = np.around(ap_results, 3)
                            ap_results_list.append(ap_results)

                            print("Model", params.model_name, "seed", params.seed, "n_trees", n_trees, "\theight",
                                    height, "\twindow_size", window_size, "Weights scores", weight_scores, "Window Already Seen",
                                    window_already_seen, "ROC-AUC: {}".format(roc_auc), "AP: {}".format(ap_results))

                            if roc_auc < min_roc_value_early_stopping:
                                break

                        if roc_auc < min_roc_value_early_stopping:
                            continue

                        all_scores_and_labels = np.array(all_scores_and_labels)
                        thresholds = []

                        for all_scores_and_labels_seed in all_scores_and_labels:
                            clf = DecisionTreeClassifier(random_state=0, max_depth=1, criterion="gini").fit(
                                X=all_scores_and_labels_seed[:, 0].reshape(-1, 1), y=all_scores_and_labels_seed[:, 1])
                            print("Thresh", clf.tree_.threshold)

                            threshold = clf.tree_.threshold[0]
                            thresholds.append(threshold)

                        f1_list = []
                        acc_list = []
                        for i, x in enumerate(all_scores_and_labels):
                            tp = 0.0
                            tn = 0.0
                            fp = 0.0
                            fn = 0.0
                            for idx, y in enumerate(x):
                                score = y[0]
                                l = y[1]
                                
                                if (score > thresholds[i] and l == 1):
                                    tp += 1
                                elif (score <= thresholds[i] and l == 0):
                                    tn += 1
                                elif (score > thresholds[i] and l == 0):
                                    fp += 1
                                elif (score <= thresholds[i] and l == 1):
                                    fn += 1

                            precision = np.around(
                                (tp) / (tp + fp), 15) if (tp + fp) > 0 else 0
                            recall_tp = np.around(
                                (tp) / (tp + fn), 15) if (tp + fn) > 0 else 0
                            recall_tn = np.around(
                                (tn) / (tn + fp), 15) if (tn + fp) > 0 else 0
                            balanced_accuracy = np.around((recall_tp + recall_tn) / 2, 15)

                            f1score = np.around((2 * precision * recall_tp) / (precision + recall_tp), 15) \
                                if (precision + recall_tp) > 0 else 0

                            f1_list.append(f1score)
                            acc_list.append(balanced_accuracy)
                            # print(seeds[i], f1score, balanced_accuracy)

                        if len(roc_auc_list) == len(seeds):
                            print("Model", params.model_name, "n_trees", n_trees, "\theight", height, "\twindow_size",
                                window_size, "\tthreshold", thresholds, "Weights scores", weight_scores, "Window Already Seen",
                                window_already_seen)

                            print("ROC: {} +- {}".format(np.mean(roc_auc_list), np.std(roc_auc_list)))
                            print("AP: {} +- {}".format(np.mean(ap_results_list), np.std(ap_results_list)))
                            print( "F1: {} +- {}\nBalancedAcc: {} +- {}".format(
                                np.mean(f1_list), np.std(f1_list), np.mean(acc_list), np.std(acc_list)))
                            print("Timing Mean: {}".format(np.mean(timings)))

import torch
from tqdm import tqdm
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from river import metrics
import torch_geometric
from torch_geometric.loader import NeighborLoader

def run_stream(params, times, df, g, model, weights, anomaly_edge_model,
               anomaly_node_model, device, window_already_seen=100, start_test_index=-1, threshold=0.5, evaluate_cyber_attacks=False):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    scores_and_labels = []
    roc_auc_over_time = []

    #roc_auc_over_time_metric = metrics.ROCAUC(n_thresholds=20)

    model.eval()

    if g.x is not None:
        x = g.x.to(device)
    edge_index = g.edge_index.cpu().numpy()#g.edge_index.to(device)
    edge_dict = g.edge_dict

    emb = model.encode(x=x.to(device), edge_index=g.edge_index.to(device)).cpu().detach()

    n_chunk_division = 30
    time_limit = 100

    if params.dataset == "UNSW-NB15":
        n_chunk_division = 30
        time_limit = 30#10
    elif params.dataset == "ISCX":
        n_chunk_division = 100
        time_limit = 30#20
    elif params.dataset == "IDS2018":
        n_chunk_division = 30
        time_limit = 240#120
    elif params.dataset == "CTU-13-Scenario-1":
        n_chunk_division = 1024
        time_limit = 300#200#100
    elif params.dataset == "CTU-13-Scenario-10":
        n_chunk_division = 1024
        time_limit = 300#200#100
    elif params.dataset == "CTU-13-Scenario-13":
        n_chunk_division = 1024
        time_limit = 450#150

    if params.dataset == "DARPA" and "DARPA_CL" in params.model_save_path:
        n_chunk_division = 90
    elif params.dataset == "IDS2018" and "IDS2018_CL" in params.model_save_path:
        n_chunk_division = 90
    elif params.dataset == "UNSW-NB15" and "UNSW-NB15_CL" in params.model_save_path:
        n_chunk_division = 180
    elif params.dataset == "ISCX" and "ISCX_CL" in params.model_save_path:
        n_chunk_division = 200

    if "CL" in params.model_save_path:
        time_limit = 1000


    n_chunk = times.shape[0] // n_chunk_division
    splitted = np.array_split(times, n_chunk)

    scalability_analysis = False
    finished_scalability_analysis = False
    num_edges_scalability = 10**4
    count_edges = 0

    time_track = tqdm(splitted, ncols=0, dynamic_ncols=False)

    edges = []
    indexes = []
    attack_names = []
    counts_for_each_attack = {}

    for t in splitted:
        edges.append(df[df.t.isin(t)][['s', 'd', 'l']].to_numpy())
        indexes.append(list(df[df.t.isin(t)].index))
        if params.dataset in ["DARPA", "UNSW-NB15"]:
            attack_names.extend(df[df.t.isin(t)]['name'].to_numpy())

    if "retrain" in vars(params) and params.retrain == 1:
        optimizer = torch.optim.Adam(
            list(model.parameters()), lr=params.learning_rate)

    total_time = time.time()

    for epoch, t in enumerate(time_track):

        if start_test_index == -1 and time.time() - total_time > time_limit:
            return 0, 0, 1, 0, [[0.5, 0], [0.5, 1]], -1, [], 0

        new_edges = edges[epoch]
        new_indexes = indexes[epoch]

        if epoch % window_already_seen == 0:
            already_seen = {}

        seen_here = {}
        new_edge_indexes = []
        for new_edge in new_edges:

            ne_edg = (new_edge[0], new_edge[1])

            if ne_edg not in edge_dict.keys():
                new_edge_indexes.append(ne_edg)
                edge_dict[ne_edg] = edge_index.shape[0] - 1 + len(new_edge_indexes)

            if scalability_analysis:
                count_edges += 1
                if count_edges == num_edges_scalability:
                    break

        if len(new_edge_indexes) > 0:
            new_edge_indexes = np.array(new_edge_indexes).T

            edge_index = np.concatenate((edge_index, new_edge_indexes), axis=1)
            edge_index_tensor = torch.LongTensor(edge_index).to(device)

            emb = model.encode(x=x, edge_index=edge_index_tensor)
            emb = emb.cpu().detach()

        if params.phi == "mean":
            edge_embs = (emb[new_edges[:, 0]] + emb[new_edges[:, 1]]) / 2
        else:
            edge_embs = (emb[new_edges[:, 1]] - emb[new_edges[:, 0]])

        if type(edge_embs) != np.ndarray:
            edge_embs = edge_embs.numpy()

        if weights[1] > 0.0:
            s_embs = emb[new_edges[:, 0]]
            d_embs = emb[new_edges[:, 1]]
        labels = new_edges[:, 2]

        if len(edge_embs.shape) == 1:
            edge_embs = edge_embs.reshape(1, -1)
            if weights[1] > 0.0:
                s_embs = s_embs.reshape(1, -1)
                d_embs = d_embs.reshape(1, -1)

        for e, edge_emb in enumerate(edge_embs):
            l = labels[e].item()
            index = new_indexes[e]
            ne_edg = (new_edges[e, 0].item(), new_edges[e, 1].item())

            if ne_edg in already_seen.keys():
                score = already_seen[ne_edg]
            else:

                features_edge = {}
                features_s = {}
                features_d = {}
                for i in range(params.out_channels):
                    features_edge[i] = edge_emb[i]
                    if weights[1] > 0.0:
                        features_s[i] = s_embs[e][i]
                        features_d[i] = d_embs[e][i]

                if params.update_hst:
                    anomaly_edge_model['MinMaxScaler'].learn_one(features_edge)
                    score_edge = anomaly_edge_model.score_and_learn_one(
                            features_edge)
                else:
                    score_edge = anomaly_edge_model.score_one(features_edge)

                if weights[1] > 0.0:

                    if params.update_hst:
                        anomaly_node_model['MinMaxScaler'].learn_one(
                            features_s)
                        anomaly_node_model['MinMaxScaler'].learn_one(
                            features_d)

                        score_s = anomaly_node_model.score_and_learn_one(
                            features_s)
                        score_d = anomaly_node_model.score_and_learn_one(
                            features_d)
                    else:
                        score_s = anomaly_node_model.score_one(features_s)
                        score_d = anomaly_node_model.score_one(features_d)

                    score = score_edge * \
                        weights[0] + score_s * \
                            weights[1] + score_d * weights[2]

                else:
                    score = score_edge

                already_seen[ne_edg] = score

            if not scalability_analysis:
                if (start_test_index != -1 and index >= start_test_index):
                    if start_test_index == index:
                        scores_and_labels = np.array(scores_and_labels)
                        print(index, ne_edg, l)
                        print(roc_auc_score(scores_and_labels[:, 1], scores_and_labels[:, 0]))
                        scores_and_labels = []
                        already_seen = {}
                        total_time = time.time()

            scores_and_labels.append((score, l))

            ## Scalablity analysis
            if scalability_analysis and count_edges == num_edges_scalability:
                finished_scalability_analysis = True
                break

        if scalability_analysis and finished_scalability_analysis:
            break

    print("Total time", time.time() - total_time)

    if params.dataset == "DARPA":
        dos_cyber_attacks = ["apache2", "back", "mailbomb", "neptune", "pingofdeath", "processtable"
                             "smurf", "syslogd", "udp-storm", "land", "teardrop"]
        r2l_cyber_attacks = ["dictionary", "ftp-write", "guest", "http-tunnel", "phf",
                             "xlock", "xsnoop", "imap", "named", "sendmail", "snmpgetattack", "snmpguess"]
        probe_cyber_attacks = ["ipsweep", "mscan", "nmap", "saint", "satan"]
        u2r_cyber_attacks = ["loadmodule", "perl", "xterm", "eject", "ffbconfig", "fdformat", "ps"]

        counts_for_each_attack = {'dos': [0, 0, 0, 0, [], []], 'probe': [0, 0, 0, 0, [], []],
                                  'r2l': [0, 0, 0, 0, [], []], 'u2r': [0, 0, 0, 0, [], []]}

    elif params.dataset == "UNSW-NB15":

        counts_for_each_attack = {'Exploits': [0, 0, 0, 0, [], []], 'Reconnaissance': [0, 0, 0, 0, [], []],
                                  'DoS': [0, 0, 0, 0, [], []], 'Shellcode': [0, 0, 0, 0, [], []],
                                  'Fuzzers': [0, 0, 0, 0, [], []], 'Worms': [0, 0, 0, 0, [], []],
                                  'Backdoors': [0, 0, 0, 0, [], []]}
    for i, (score, l) in enumerate(scores_and_labels):

        if params.dataset in ["DARPA", "UNSW-NB15"] and evaluate_cyber_attacks:

            attack_name = attack_names[i]

            if params.dataset == "DARPA":
                if (score > threshold and l == 1):
                    if attack_name in dos_cyber_attacks:
                        counts_for_each_attack['dos'][0] += 1
                    elif attack_name in probe_cyber_attacks:
                        counts_for_each_attack['probe'][0] += 1
                    elif attack_name in r2l_cyber_attacks:
                        counts_for_each_attack['r2l'][0] += 1
                    elif attack_name in u2r_cyber_attacks:
                        counts_for_each_attack['u2r'][0] += 1
                elif (score <= threshold and l == 0):
                    if attack_name in dos_cyber_attacks:
                        counts_for_each_attack['dos'][1] += 1
                    elif attack_name in probe_cyber_attacks:
                        counts_for_each_attack['probe'][1] += 1
                    elif attack_name in r2l_cyber_attacks:
                        counts_for_each_attack['r2l'][1] += 1
                    elif attack_name in u2r_cyber_attacks:
                        counts_for_each_attack['u2r'][1] += 1
                elif (score > threshold and l == 0):
                    if attack_name in dos_cyber_attacks:
                        counts_for_each_attack['dos'][2] += 1
                    elif attack_name in probe_cyber_attacks:
                        counts_for_each_attack['probe'][2] += 1
                    elif attack_name in r2l_cyber_attacks:
                        counts_for_each_attack['r2l'][2] += 1
                    elif attack_name in u2r_cyber_attacks:
                        counts_for_each_attack['u2r'][2] += 1
                elif (score <= threshold and l == 1):
                    if attack_name in dos_cyber_attacks:
                        counts_for_each_attack['dos'][3] += 1
                    elif attack_name in probe_cyber_attacks:
                        counts_for_each_attack['probe'][3] += 1
                    elif attack_name in r2l_cyber_attacks:
                        counts_for_each_attack['r2l'][3] += 1
                    elif attack_name in u2r_cyber_attacks:
                        counts_for_each_attack['u2r'][3] += 1

                if attack_name in dos_cyber_attacks:
                    counts_for_each_attack['dos'][4].append(l)
                    counts_for_each_attack['dos'][5].append(score)
                elif attack_name in probe_cyber_attacks:
                    counts_for_each_attack['probe'][4].append(l)
                    counts_for_each_attack['probe'][5].append(score)
                elif attack_name in r2l_cyber_attacks:
                    counts_for_each_attack['r2l'][4].append(l)
                    counts_for_each_attack['r2l'][5].append(score)
                elif attack_name in u2r_cyber_attacks:
                    counts_for_each_attack['u2r'][4].append(l)
                    counts_for_each_attack['u2r'][5].append(score)

            elif params.dataset == "UNSW-NB15" and attack_name in counts_for_each_attack.keys():
                counts_for_each_attack[attack_name][4].append(l)
                counts_for_each_attack[attack_name][5].append(score)

        if (score > threshold and l == 1):
            tp += 1
        elif (score <= threshold and l == 0):
            tn += 1
        elif (score > threshold and l == 0):
            fp += 1
        elif (score <= threshold and l == 1):
            fn += 1


    return tp, tn, fp, fn, scores_and_labels, time.time() - total_time, roc_auc_over_time, counts_for_each_attack

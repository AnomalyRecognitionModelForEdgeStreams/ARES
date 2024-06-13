import torch
from tqdm import tqdm
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from river import metrics


def run_stream(params, times, df, g, model, weights, anomaly_edge_model,
               anomaly_node_model, device, window_already_seen=100, start_test_index=-1, threshold=0.5):
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

    n_chunk = times.shape[0] // n_chunk_division
    splitted = np.array_split(times, n_chunk)

    scalability_analysis = False
    finished_scalability_analysis = False
    num_edges_scalability = 10**4
    count_edges = 0

    time_track = tqdm(splitted, ncols=0, dynamic_ncols=False)

    edges = [] #df[df.t.isin(t)][['s', 'd', 'l']].to_numpy()
    indexes = [] #list(df[df.t.isin(t)].index)

    for t in splitted:
        edges.append(df[df.t.isin(t)][['s', 'd', 'l']].to_numpy())
        indexes.append(list(df[df.t.isin(t)].index))

    total_time = time.time()

    for epoch, t in enumerate(time_track):

        if start_test_index == -1 and time.time() - total_time > time_limit:
            return 0, 0, 1, 0, [[0.5, 0], [0.5, 1]], -1, []

        new_edges = edges[epoch]
        new_indexes = indexes[epoch]

        if epoch % window_already_seen == 0:
            already_seen = {}

        new_edge_indexes = []
        for new_edge in new_edges:

            ne_edg = (new_edge[0], new_edge[1])
            #new_edge_indexes.append(ne_edg)

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

        emb = model.encode(x=x, edge_index=torch.LongTensor(edge_index).to(device)).cpu().detach()

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
    for score, l in scores_and_labels:
        if (score > threshold and l == 1):
            tp += 1
        elif (score <= threshold and l == 0):
            tn += 1
        elif (score > threshold and l == 0):
            fp += 1
        elif (score <= threshold and l == 1):
            fn += 1


    return tp, tn, fp, fn, scores_and_labels, time.time() - total_time, roc_auc_over_time

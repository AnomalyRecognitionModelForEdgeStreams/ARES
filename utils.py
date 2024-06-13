import time
import pandas as pd
import numpy as np
import torch
import os
import os
import torch_geometric

def load_dataset(dataset):
    processed_path = "data/processed/{}".format(dataset)
    snapshots = []

    for i in range(len(os.listdir(processed_path))):
        snapshots.append(torch.load("{}/{}.pt".format(processed_path, i)))

    return snapshots


def preprocess_dataset(params):

    dataset = params.dataset
    perc_train = params.perc_train
    perc_val = params.perc_val

    processed_path = "data/processed/{}".format(dataset)

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    if dataset == "DARPA":
        path = "data/processed/DARPA/darpa_original.csv"
        df = pd.read_csv(path, header=None, names=["s", "d", "t", "l"])

        df.l = df.l != '-'
        df.l = df.l.astype('int')

        all_site = pd.concat([df.s, df.d])
        all_site = all_site.astype('category')
        all_site = all_site.cat.codes
        df.s = all_site[:df.shape[0]]
        df.d = all_site[df.shape[0]:]
        df.t = df.t.astype('category')
        df.t = df.t.cat.codes + 1  # Time starts from 1

        train_chunk = int(df.shape[0] * perc_train)
        train_chunk_t = df.iloc[train_chunk].t

        train_chunk_tmp = int(df.shape[0] * 0.6)

        val_chunk = int(df.shape[0] * perc_val) + train_chunk_tmp
        val_chunk_t = df.iloc[val_chunk].t

        train = df[df.t <= train_chunk_t]
        val_and_train = df[df.t <= val_chunk_t]
        val_and_train = val_and_train[val_and_train['l'] == 0]
        val = df[(df.t > train_chunk_t) & (df.t <= val_chunk_t)]
        test = df[df.t > val_chunk_t]
        val_and_test = df[df.t > train_chunk_t]

        val = val.reset_index()
        test = test.reset_index()

        max_node = max(df.s.max(), df.d.max()) + 1

        x = torch.arange(max_node,
                         dtype=torch.float32).reshape(-1,
                                                      1) / (max_node - 1)
        x = torch.FloatTensor(
            [[x_u for _ in range(params.in_channels)] for x_u in x])

        w_train = train[['s', 'd']].value_counts()
        w_train = w_train.reset_index()
        w_train['i'] = np.arange(w_train.shape[0])
        w_train_dict = w_train.set_index(["s", "d"]).to_dict()['i']

        train_graph = torch_geometric.data.Data(x=x, edge_index=torch.LongTensor(
            w_train[['s', 'd']].to_numpy()).t().contiguous(), edge_dict=w_train_dict).cpu()

        w_val = val[['s', 'd']].value_counts()
        w_val = w_val.reset_index()
        w_val['i'] = np.arange(w_val.shape[0])
        w_val_dict = w_val.set_index(["s", "d"]).to_dict()['i']

        val_graph = torch_geometric.data.Data(x=x, edge_index=torch.LongTensor(
            w_val[['s', 'd']].to_numpy()).t().contiguous(), edge_dict=w_val_dict).cpu()


        return train_graph, val_graph, train, val, test, val_and_test, df


def save_ctu(dataset):
    df = pd.read_csv("data/raw/{}/data.csv".format(dataset))

    df['l'] = df['Label'].apply(
        lambda x: 1 if x.startswith("flow=From-Botnet") else 0)

    df = df[['SrcAddr', 'DstAddr', 'StartTime', 'l']]

    df['s'] = df['SrcAddr'].astype(str)
    df['d'] = df['DstAddr'].astype(str)
    df['t'] = df['StartTime']

    df = df[['s', 'd', 't', 'l']]

    nodes = {}
    timings = {}
    count_nodes = 0
    count_timings = 0

    df = df.sort_values('t')
    new_df = []
    for r in df.iloc:
        if r.s not in nodes.keys():
            nodes[r.s] = count_nodes
            nodes[r.s] = count_nodes
            count_nodes += 1

        if r.d not in nodes.keys():
            nodes[r.d] = count_nodes
            count_nodes += 1

        if r.t not in timings.keys():
            timings[r.t] = count_timings
            count_timings += 1

        new_df.append([nodes[r.s], nodes[r.d], timings[r.t], r.l])

    df = pd.DataFrame(new_df, columns=['s', 'd', 't', 'l'])

    print("data/processed/{}".format(dataset))
    if not os.path.exists("data/processed/{}".format(dataset)):
        os.makedirs("data/processed/{}".format(dataset))

    f = open("data/processed/{}/processed.csv".format(dataset), "w")
    for t in df.to_numpy():
        f.write("{},{},{}\n".format(t[0], t[1], t[2]))
    f.close()

    f = open("data/processed/{}/ground_truth.csv".format(dataset), "w")
    for t in df.to_numpy():
        f.write("{}\n".format(t[3]))
    f.close()

    f = open("data/processed/{}/shape.txt".format(dataset), "w")
    f.write("{}".format(df.shape[0]))
    f.close()

    print("Saved!")
    exit(0)


def load_data(params):
    perc_train = params.perc_train
    perc_val = params.perc_val
    in_channels = params.in_channels
    dataset = params.dataset

    df = pd.read_csv(
        "data/processed/{}/processed.csv".format(dataset),
        header=None,
        names=['s', 'd', 't'])
    df_gt = pd.read_csv(
        "data/processed/{}/ground_truth.csv".format(dataset),
        header=None,
        names=['l'])

    df['l'] = df_gt['l']

    train_chunk = int(df.shape[0] * perc_train)
    train_chunk_t = df.iloc[train_chunk].t

    train_chunk_tmp = int(df.shape[0] * 0.6)

    val_chunk = int(df.shape[0] * perc_val) + train_chunk_tmp
    val_chunk_t = df.iloc[val_chunk].t

    train = df[df.t <= train_chunk_t]
    val_and_train = df[df.t <= val_chunk_t]
    val_and_train = val_and_train[val_and_train['l'] == 0]
    val = df[(df.t > train_chunk_t) & (df.t <= val_chunk_t)]
    test = df[df.t > val_chunk_t]
    val_and_test = df[df.t > train_chunk_t]

    val = val.reset_index()
    test = test.reset_index()

    max_node = max(df.s.max(), df.d.max()) + 1

    x = torch.arange(max_node,
                     dtype=torch.float32).reshape(-1,
                                                  1) / (max_node - 1)
    x = torch.FloatTensor([[x_u for _ in range(in_channels)] for x_u in x])

    w_train = train[['s', 'd']].value_counts()
    w_train = w_train.reset_index()
    w_train['i'] = np.arange(w_train.shape[0])
    w_train_dict = w_train.set_index(["s", "d"]).to_dict()['i']

    train_graph = torch_geometric.data.Data(x=x, edge_index=torch.LongTensor(
        w_train[['s', 'd']].to_numpy()).t().contiguous(), edge_dict=w_train_dict).cpu()

    w_val = val[['s', 'd']].value_counts()
    w_val = w_val.reset_index()
    w_val['i'] = np.arange(w_val.shape[0])
    w_val_dict = w_val.set_index(["s", "d"]).to_dict()['i']

    val_graph = torch_geometric.data.Data(x=x, edge_index=torch.LongTensor(
        w_val[['s', 'd']].to_numpy()).t().contiguous(), edge_dict=w_val_dict).cpu()

    return train_graph, val_graph, train, val, test, val_and_test, df


def save_unsw():
    columns = [
        'srcip',
        "sport",
        "dstip",
        "dsport",
        "proto",
        "state",
        "dur",
        "sbytes",
        "dbytes",
        "sttl",
        "dttl",
        "sloss",
        "dloss",
        "service",
        "Sload",
        "Dload",
        "Spkts",
        "Dpkts",
        "swin",
        "dwin",
        "stcpb",
        "dtcpb",
        "smeansz",
        "dmeansz",
        "trans_depth",
        "res_bdy_len",
        "Sjit",
        "Djit",
        "Stime",
        "Ltime",
        "Sintpkt",
        "Dintpkt",
        "tcprtt",
        "synack",
        "ackdat",
        "is_sm_ips_ports",
        "ct_state_ttl",
        "ct_flw_http_mthd",
        "is_ftp_login",
        "ct_ftp_cmd",
        "ct_srv_src",
        "ct_srv_dst",
        "ct_dst_ltm",
        "ct_src_ltm",
        "ct_src_dport_ltm",
        "ct_dst_sport_ltm",
        "ct_dst_src_ltm",
        "attack_cat",
        "Label"]

    df1 = pd.read_csv(
        "data/raw/UNSW-NB15/UNSW-NB15_1.csv",
        header=None,
        names=columns)
    df2 = pd.read_csv(
        "data/raw/UNSW-NB15/UNSW-NB15_2.csv",
        header=None,
        names=columns)
    df3 = pd.read_csv(
        "data/raw/UNSW-NB15/UNSW-NB15_3.csv",
        header=None,
        names=columns)
    df4 = pd.read_csv(
        "data/raw/UNSW-NB15/UNSW-NB15_4.csv",
        header=None,
        names=columns)

    df = pd.concat((df1, df2, df3, df4))

    df = df[['srcip', 'sport', 'dstip', 'dsport', "Ltime", 'Label']]

    # + np.full(df.shape[0], ":") + df['sport'].astype(str)
    df['s'] = df['srcip'].astype(str)
    # + np.full(df.shape[0], ":") + df['dsport'].astype(str)
    df['d'] = df['dstip'].astype(str)
    df['t'] = df['Ltime']
    df['l'] = df['Label']

    df = df[['s', 'd', 't', 'l']]

    nodes = {}
    timings = {}
    count_nodes = 0
    count_timings = 0

    df = df.sort_values('t')
    new_df = []
    for r in df.iloc:
        if r.s not in nodes.keys():
            nodes[r.s] = count_nodes
            nodes[r.s] = count_nodes
            count_nodes += 1

        if r.d not in nodes.keys():
            nodes[r.d] = count_nodes
            count_nodes += 1

        if r.t not in timings.keys():
            timings[r.t] = count_timings
            count_timings += 1

        new_df.append([nodes[r.s], nodes[r.d], timings[r.t], r.l])

    df = pd.DataFrame(new_df, columns=['s', 'd', 't', 'l'])

    f = open("data/processed/UNSW-NB15/processed.csv", "w")
    for t in df.to_numpy():
        f.write("{},{},{}\n".format(t[0], t[1], t[2]))
    f.close()

    f = open("data/processed/UNSW-NB15/ground_truth.csv", "w")
    for t in df.to_numpy():
        f.write("{}\n".format(t[3]))
    f.close()

    f = open("data/processed/UNSW-NB15/shape.txt", "w")
    f.write("{}".format(df.shape[0]))
    f.close()

    print("Saved!")
    exit(0)

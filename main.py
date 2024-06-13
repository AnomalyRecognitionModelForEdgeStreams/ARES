import time
import os
import argparse
import json
import sys

with open(sys.argv[1]) as fp:
    params = json.load(fp)

sys.argv = sys.argv[:1]
parser = argparse.ArgumentParser()
t_args = argparse.Namespace()
t_args.__dict__.update(params)
params = parser.parse_args(args=None, namespace=t_args)

os.environ['CUDA_VISIBLE_DEVICES']= "{}".format(str(params.GPU))

import torch

from pipeline import pipeline_validation, pipeline_test
from utils import preprocess_dataset, save_unsw, save_ctu, load_data

from trainer import Trainer
import numpy as np
import torch_geometric



if params.save_files:
    if params.dataset == "UNSW-NB15":
        train_graph, val_graph, val, test, loaders = save_unsw()
    if params.dataset.startswith("CTU-13-Scenario"):
        train_graph, val_graph, val, test, loaders = save_ctu(
            params.dataset)

if params.dataset == "DARPA":
    train_graph, val_graph, train, val, test, val_and_test, df = preprocess_dataset(
        params)
else:
    train_graph, val_graph, train, val, test, val_and_test, df = load_data(
        params)


GPU = params.GPU
device_string = 'cuda:{}'.format("0") if torch.cuda.is_available() and GPU != "cpu" else 'cpu'
device = torch.device(device_string)

params.device = device

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

if not os.path.exists("plot_loss"):
    os.makedirs("plot_loss")

trainer = Trainer(params)

if params.training:
    trainer.fit(train_graph, val_graph)
else:

    model = trainer.model
    model = model.to(device)
    model.load_state_dict(
        torch.load(
            "{}.pt".format(
                params.model_save_path),
            map_location=device))

    max = 1
    min = 0

    edge_embeddings_stream = []
    edges = []

    limits = {}
    for i in range(params.out_channels):
        limits[i] = (min, max)

    if params.validation:

        weights_scores = [(1.0, 0.0, 0.0), (0.33, 0.33, 0.33)]
        window_already_seen = [64, 32, 16, 8, 4]
        n_trees_list = [8, 16, 32, 64]
        height_list = [3, 6, 9, 12]
        window_size_list = [8, 64, 512, 1024, 2048, 4096, 8192]
        seeds = [
            42,
            1774956081,
            1412186673,
            32879431,
            1915539092,
            249573454,
            38938983,
            487190043,
            1989860718]

        g = train_graph.clone()
        times = val.t.unique()

        x = g.x.to(device)
        edge_index = g.edge_index.to(device)

        emb = model.encode(x=x, edge_index=edge_index).cpu().detach()

        pipeline_validation(
            n_trees_list,
            height_list,
            window_size_list,
            weights_scores,
            seeds,
            window_already_seen,
            params,
            limits,
            times,
            val,
            g,
            model,
            device,
            emb)

    else:

        n_trees = params.n_trees
        height = params.height
        window_size = params.window_size
        thresholds = params.thresholds
        thresholds = np.array(thresholds.split(", ")).astype(np.float32)
        window_already_seen = params.window_already_seen
        seeds = [
            42,
            1774956081,
            1412186673,
            32879431,
            1915539092,
            249573454,
            38938983,
            487190043,
            1989860718]
        compute_mean_std = True

        weights = params.weights.split(",")
        weights = [float(w) for w in weights]

        output_file = "log_test_{}_{}".format(
            params.dataset, params.model_name)
        if params.update_hst:
            output_file += "_updateHST"
        output_file += ".txt"

        g = train_graph.clone()
        times = np.concatenate((val.t.unique(), test.t.unique()))

        x = g.x.to(device)
        edge_index = g.edge_index.to(device)

        emb = model.encode(x=x, edge_index=edge_index).cpu().detach()

        pipeline_test(
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
            times,
            val_and_test,
            g,
            model,
            device,
            emb)

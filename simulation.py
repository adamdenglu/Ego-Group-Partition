# -*- coding: utf-8 -*-
# Copyright 2024 Tencent Inc.  All rights reserved.
# Author: adamdeng@tencent.com
import os
import numpy as np
import pandas as pd
import networkx as nx
from time import time
from joblib import Parallel, delayed
from src.ego_cluster import EgoCluster
from src.ego_group_partition import EgoGroupPartition
from src.simulation_utils import simulation


def run_simulation(ego_object, ego_cluster, name, rep=100, model='linear'):

    est, tau = simulation(ego_object, ego_cluster, rep, model)
    # save results
    path = "./results/"
    if not os.path.exists(path):
        os.makedirs(path)
    rnd = int(time() * 1e8 % 1e8)
    save_path = path + f"{model}_results_{rnd}.npz"
    np.savez(
        save_path,
        # estimation results
        est=est,
        # true te
        tau=tau,
        # set up
        model=model,
        name=name
    )


if __name__ == "__main__":
    path = "./dataset/socfb-Cornell5.mtx"
    df_graph = pd.read_table(
        path, skiprows=1, names=["source", "target"], sep=" "
    )
    graph = nx.from_pandas_edgelist(df_graph)

    # ego cluster
    ego_cluster = EgoCluster(
        graph.copy(), loss_rates_threshold=0.75, num_bins=500
    )
    ego_cluster.clustering()
    print("number of selected egos :", len(ego_cluster._ego_clusters.keys()))
    ego_ratio = len(ego_cluster._ego_clusters.keys()) / graph.number_of_nodes()
    print(f"ego ratio : {ego_ratio}")

    rep = 100
    models = ["linear", "convex"]
    name = "ego_cluster"
    begin_time = time()
    Parallel(n_jobs=-1, verbose=5)(
        delayed(run_simulation)(ego_cluster, True, name, rep, model)
        for model in models for s in range(10)
    )
    print(f"Finished ego cluster in {time() - begin_time} seconds.")

    # ego group partition 1
    egp1 = EgoGroupPartition(graph.copy(), ego_ratio=ego_ratio, threshold=0)
    rep = 100
    models = ["linear", "convex"]
    name = "egp threshold 0"
    begin_time = time()
    Parallel(n_jobs=-1, verbose=5)(
        delayed(run_simulation)(egp1, False, name, rep, model)
        for model in models for s in range(10)
    )
    print(f"Finished in {time() - begin_time} seconds.")

    # ego group partition 2
    egp2 = EgoGroupPartition(graph.copy(), ego_ratio=ego_ratio, threshold=0.2)
    rep = 100
    models = ["linear", "convex"]
    name = "egp threshold 0.2"
    begin_time = time()
    Parallel(n_jobs=-1, verbose=5)(
        delayed(run_simulation)(egp2, False, name, rep, model)
        for model in models for s in range(10)
    )
    print(f"Finished in {time() - begin_time} seconds.")

    # ego group partition 3
    egp3 = EgoGroupPartition(graph.copy(), ego_ratio=ego_ratio, threshold=0.5)
    rep = 100
    models = ["linear", "convex"]
    name = "egp threshold 0.5"
    begin_time = time()
    Parallel(n_jobs=-1, verbose=5)(
        delayed(run_simulation)(egp3, False, name, rep, model)
        for model in models for s in range(10)
    )
    print(f"Finished in {time() - begin_time} seconds.")

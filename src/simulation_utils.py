# -*- coding: utf-8 -*-
# Copyright 2024 Tencent Inc.  All rights reserved.
# Author: adamdeng@tencent.com
import numpy as np
import networkx as nx
from src.ego_group_partition import EgoGroupPartition


def my_convex_func(x):
    '''
    Convex function: y = - exp(-3x) + 1
    '''
    ans = -np.exp(-3 * x) + 1
    return ans


def network_data_generator(graph, model="linear"):
    '''
    Generate data directed from the adjacency matrix.

    Parameters
    ----------
    graph : networkx object
        networkx object representing the graph

    '''
    # generate the potential outcomes
    nx.set_node_attributes(graph, 0, "y")
    if model == "linear":
        for node in graph.nodes:
            graph.nodes[node]["y"] = (
                1 + 1 * graph.nodes[node]["z"] +
                1 * sum(
                    [graph.nodes[ngbr]["z"] for ngbr in graph[node]]
                )/graph.degree[node] +
                1 * np.random.normal(loc=0, scale=1)
            )
    elif model == "convex":
        for node in graph.nodes:
            graph.nodes[node]["y"] = (
                1 + 1 * graph.nodes[node]["z"] +
                my_convex_func(sum(
                    [graph.nodes[ngbr]["z"] for ngbr in graph[node]]
                )/graph.degree[node]) +
                1 * np.random.normal(loc=0, scale=1)
            )


def estimate(graph):
    '''
    Calculte the estimators given graph and probability of treatment.

    Parameters
    ----------
    graph : networkx object
        networkx object representing the graph

    Returns
    ----------
    est : float
        estimator for the global average treatment effect

    '''
    treat_ego = []
    ctrl_ego = []
    for node in graph.nodes:
        if graph.nodes[node]["is_ego"] == 0:
            continue
        if graph.nodes[node]["z"] == 1:
            treat_ego.append(graph.nodes[node]["y"])
        else:
            ctrl_ego.append(graph.nodes[node]["y"])
    est = np.mean(treat_ego) - np.mean(ctrl_ego)

    return est


def simulation(ego_object, ego_cluster=False, reps=100, model="linear"):
    '''
    Returns the estimators for simulated cases for 'reps' times with a fixed
    graph once generated

    Parameters
    ----------
    graph : str/networkx
        a str representing the type of graph or a networkx object
    reps : int
        simulation for 'reps' time
    model : str
        set the generating distribution of the outcomes
    '''
    est = np.zeros(reps)

    if model == "linear":
        tau = 1 + 1
    elif model == "convex":
        tau = 1 + my_convex_func(1) - my_convex_func(0)

    for i in range(reps):
        if ego_cluster:
            ego_object.random()
            network_data_generator(ego_object.graph, model)
            est[i] = estimate(ego_object.graph)
        else:
            egp = EgoGroupPartition(
                ego_object.graph.copy(), ego_ratio=ego_object.ego_ratio,
                threshold=ego_object.threshold
            )
            egp.partition()
            network_data_generator(egp.graph, model)
            est[i] = estimate(egp.graph)

    return est, tau

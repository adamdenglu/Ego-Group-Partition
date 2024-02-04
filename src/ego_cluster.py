# -*- coding: utf-8 -*-
# Copyright 2024 Tencent Inc.  All rights reserved.
# Author: adamdeng@tencent.com
import math
import numpy as np
import networkx as nx


class EgoCluster(object):
    '''Ego clustering procedure.'''
    def __init__(
        self, graph, loss_rates_threshold=0.15, num_bins=5
    ):
        '''Initialize the algorithm with the loss rates threshold.'''
        self.graph = graph
        self._loss_rates_threshold = loss_rates_threshold
        self._num_bins = num_bins

    def _classify2bins(self):
        '''Classify nodes into degree bins.'''
        degree_to_nodes = {}
        for node in self.graph.nodes():
            degree = self.graph.degree[node]
            if degree not in degree_to_nodes:
                degree_to_nodes[degree] = set()
            degree_to_nodes[degree].add(node)
        degree_values = sorted(degree_to_nodes.keys())

        num_bins = self._num_bins
        num_nodes = self.graph.number_of_nodes()
        #  numer of nodes in each bin
        bin_num_nodes = num_nodes // num_bins

        degree_bins = {}
        for iter in range(num_bins):
            degree_bins[iter] = {}
            # degree in bins
            degree_bins[iter]["degree"] = set()
            # nodes in bins
            degree_bins[iter]["nodes"] = set()
            # number of nodes in bins
            degree_bins[iter]["num_nodes"] = 0

            tmp_value = 0
            # from low degree to high degree
            while tmp_value < bin_num_nodes:
                if len(degree_values) == 0:
                    break

                degree = degree_values.pop(0)
                candidate = degree_to_nodes[degree]

                if tmp_value + len(candidate) <= bin_num_nodes:
                    tmp_value += len(candidate)
                    degree_bins[iter]["degree"].add(degree)
                    # selected_nodes = set(
                    #     data_degree[data_degree.degree == degree].fuin
                    # )
                    degree_bins[iter]["nodes"] = (
                        degree_bins[iter]["nodes"] | candidate
                    )
                    degree_bins[iter]["num_nodes"] += len(candidate)
                    # data_degree.loc[candidate_index, "picked"] = 1
                else:
                    need_pick = bin_num_nodes - tmp_value
                    selected_nodes = np.random.choice(
                        list(candidate), size=need_pick, replace=False
                    )
                    tmp_value += len(selected_nodes)
                    degree_bins[iter]["degree"].add(degree)
                    degree_bins[iter]["nodes"] = (
                        degree_bins[iter]["nodes"] | set(selected_nodes)
                    )
                    degree_bins[iter]["num_nodes"] += len(selected_nodes)
                    degree_to_nodes[degree] = (
                        degree_to_nodes[degree] - set(selected_nodes)
                    )
                    degree_values = [degree] + degree_values

        return degree_bins

    def clustering(self):
        '''Ego clustering procedure.
        '''
        degree_bins = self._classify2bins()

        # record ego and its alters
        ego_clusters = {}
        # set of selected nodes
        node_seleted = set()
        # set of selected neighbors of egos
        ego_neighbor = set()

        stop = False
        loss_rates_threshold = self._loss_rates_threshold
        loop_num = 0
        while True:
            for iter in degree_bins.keys():
                # if there is no ego in bin, stop
                if len(degree_bins[iter]["nodes"]) == 0:
                    stop = True
                    break

                selected = False
                node_candidates = (
                    degree_bins[iter]["nodes"].copy() - node_seleted
                )
                while not selected:
                    # random select one node as ego
                    if len(node_candidates) == 0:
                        stop = True
                        break
                    ego = np.random.choice(list(node_candidates))
                    node_candidates.remove(ego)
                    # all neighbors of node
                    alters = set([node for node in self.graph[ego]])
                    # all non-selected neighbors
                    non_selected_alters = alters - node_seleted
                    # compute loss rate
                    ego_loss_rate = 1 - len(non_selected_alters) / len(alters)

                    if ego_loss_rate < loss_rates_threshold:
                        # if loss rate is less than threshold, select it
                        node_seleted.add(ego)
                        ego_clusters[ego] = {}
                        ego_clusters[ego]["loop_num"] = loop_num
                        ego_clusters[ego]["bin"] = iter
                        selected_alters_num = math.ceil(
                            len(alters) * (1-loss_rates_threshold)
                        )
                        selected_alters = np.random.choice(
                            list(non_selected_alters),
                            size=selected_alters_num,
                            replace=False
                        )

                        ego_clusters[ego]["alters"] = set(selected_alters)
                        # mark the neighbors as selected
                        node_seleted = node_seleted | set(selected_alters)
                        # add selected neighbors
                        ego_neighbor = ego_neighbor | alters

                        selected = True
                    # if loss rate is larger than threshold, continue

            if stop:
                self._final_bin = iter - 1
                self._loop_num = loop_num
                break

            loop_num += 1

        # if self._drop_final_loop and self._final_bin != (self._num_bins-1):
        #     ego_set = set(ego_clusters.keys())
        #     for ego in list(ego_set):
        #         if ego_clusters[ego]["loop_num"] == self._loop_num:
        #             del ego_clusters[ego]

        ego_set = set(ego_clusters.keys())
        nx.set_node_attributes(self.graph, 0, "is_ego")
        nx.set_node_attributes(
            self.graph, {node: 1 for node in ego_set}, "is_ego"
        )

        # all non-selected neighbors of egos
        ego_neighbor_left = ego_neighbor - node_seleted
        for node in ego_neighbor_left:
            neighbors = set([x for x in self.graph[node]])
            attach_ego_candidates = neighbors & ego_set
            if len(attach_ego_candidates) == 0:
                continue
            ego_random = np.random.choice(list(attach_ego_candidates))
            ego_clusters[ego_random]["alters"] = (
                ego_clusters[ego_random]["alters"] | {node}
            )
        self._ego_clusters = ego_clusters

    def random(self):
        nx.set_node_attributes(self.graph, 0, "z")
        for ego in self._ego_clusters.keys():
            self._ego_clusters[ego]["z"] = np.random.randint(low=0, high=2)
            if self._ego_clusters[ego]["z"] == 1:
                tr_nodes = self._ego_clusters[ego]["alters"] | {ego}
                nx.set_node_attributes(
                    self.graph, {node: 1 for node in tr_nodes}, "z"
                )

        nx.set_node_attributes(self.graph, 0, "treated_neighbor_ratio")
        for ego in self._ego_clusters.keys():
            s = 0
            for node in self.graph[ego]:
                s += self.graph.nodes[node]["z"]
            self.graph.nodes[ego]["treated_neighbor_ratio"] = s / \
                self.graph.degree[ego]

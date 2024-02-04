# -*- coding: utf-8 -*-
# Copyright 2024 Tencent Inc.  All rights reserved.
# Author: adamdeng@tencent.com
import numpy as np
import networkx as nx


class EgoGroupPartition(object):
    '''Ego group partition algorithm.'''
    def __init__(self, graph, ego_ratio=0.1, threshold=1):
        '''Initialize the algorithm with the loss rates threshold.'''
        self.graph = graph
        self.ego_ratio = ego_ratio
        self.threshold = threshold

    def _assign_alter(self, alter):
        # select all ego neivghbors
        ego_neighbors = [
            node for node in self.graph[alter]
            if self.graph.nodes[node]["is_ego"] == 1
        ]

        if len(ego_neighbors) > 0:
            treat_ego = ctrl_ego = 0
            for ego in ego_neighbors:
                if self.graph.nodes[ego]["z"] == 1:
                    treat_ego += 1 / self.graph.degree[ego]
                else:
                    ctrl_ego += 1 / self.graph.degree[ego]
            if ctrl_ego == 0:
                self.graph.nodes[alter]["z"] = 1
            else:
                delta = treat_ego / ctrl_ego - 1
                if delta > self.threshold:
                    self.graph.nodes[alter]["z"] = 1
                else:
                    self.graph.nodes[alter]["z"] = 0
        else:
            self.graph.nodes[alter]["z"] = np.random.randint(low=0, high=2)

    def partition(self):
        '''Ego group partition procedure.
        '''
        # randomly select ego nodes
        nodes = np.array(list(self.graph.nodes))
        ego_index = np.random.binomial(1, self.ego_ratio, size=len(nodes))
        ego_list = list(nodes[ego_index == 1])
        nx.set_node_attributes(self.graph, 0, "is_ego")
        nx.set_node_attributes(
            self.graph, {node: 1 for node in ego_list}, "is_ego"
        )
        alter_list = list(set(nodes) - set(ego_list))

        nx.set_node_attributes(self.graph, 0, "z")
        rn = np.random.uniform(low=0.0, high=1.0, size=len(ego_list))
        tr_egos = np.array(ego_list)[np.where(rn < 0.5)[0]]
        nx.set_node_attributes(self.graph, {node: 1 for node in tr_egos}, "z")

        # begin_time = time()
        for alter in alter_list:
            self._assign_alter(alter)
        # print(f"Finished partition in {time() - begin_time} seconds.")

        nx.set_node_attributes(self.graph, 0, "treated_neighbor_ratio")
        for ego in ego_list:
            s = 0
            for node in self.graph[ego]:
                s += self.graph.nodes[node]["z"]
            self.graph.nodes[ego]["treated_neighbor_ratio"] = s / \
                self.graph.degree[ego]

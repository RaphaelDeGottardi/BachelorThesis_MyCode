
from itertools import combinations

import numpy as np
import networkx as nx


from graph_pkg_core.algorithm.graph_edit_distance import GED
from graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from graph_pkg_core.graph.edge import Edge
from graph_pkg_core.graph.graph import Graph
from graph_pkg_core.graph.label.label_edge import LabelEdge
from graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
from graph_pkg_core.graph.node import Node


def main():
    n_nodes1 = 5
    n_nodes2 = 6

    gr1 = Graph('gr1', 'gr1.gxl', n_nodes1)
    gr2 = Graph('gr2', 'gr2.gxl', n_nodes2)


    for i in range(n_nodes1):
        tmp_node = Node(i, LabelNodeVector(np.array([i, 1.])))
        gr1.add_node(tmp_node)

    for idx_start, idx_end in combinations(range(n_nodes1), 2):
        tmp_edge = Edge(idx_start, idx_end, LabelEdge(0))
        gr1.add_edge(tmp_edge)

    for i in range(n_nodes2):
        tmp_node = Node(i, LabelNodeVector(np.array([i, 1.])))
        gr2.add_node(tmp_node)

    for idx_start, idx_end in combinations(range(n_nodes2), 2):
        tmp_edge = Edge(idx_start, idx_end, LabelEdge(0))
        gr2.add_edge(tmp_edge)

    ged = GED(EditCostVector(1., 1., 1., 1., 'euclidean'))  # EditCostVector takes weights as an input

    cost = ged.compute_edit_distance(gr1, gr2, heuristic=True)

    print(cost)


if __name__ == "__main__":
    main()
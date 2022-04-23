
from itertools import combinations

import numpy as np
import networkx as nx
import os


from graph_pkg_core.algorithm.graph_edit_distance import GED
from graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from graph_pkg_core.graph.edge import Edge
from graph_pkg_core.graph.graph import Graph
from graph_pkg_core.graph.label.label_edge import LabelEdge
from graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
from graph_pkg_core.graph.node import Node
from graph_pkg_core.loader.loader_vector import LoaderVector


def main():
    n_nodes1 = 5
    n_nodes2 = 6

    loader_vector = LoaderVector('/home/raphaeldegottardi/BachelorThesis/MyCode/HashedGraphs_P', use_wl_attr=True)
    graphs = loader_vector.load()
    print(graphs[1])
    print(graphs[2])


    ged = GED(EditCostVector(1., 1., 1., 1., 'euclidean', use_wl_attr=True))  # EditCostVector takes weights as an input

    cost = ged.compute_edit_distance(graphs[1], graphs[2], heuristic=True)

    print(cost)


if __name__ == "__main__":
    main()
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
    
    G1 = nx.Graph()
    G1.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)])
    G2 = nx.Graph()
    G2.add_edges_from([(1, 4), (2, 4), (1, 6), (1, 5), (4, 6)])
    g1_hashes = nx.weisfeiler_lehman_subgraph_hashes(G1, iterations=3, digest_size=8)
    g2_hashes = nx.weisfeiler_lehman_subgraph_hashes(G2, iterations=3, digest_size=8)

    print(g1_hashes[1])
    # print(g2_hashes[5])

    #print(G2.degree)

if __name__ == "__main__":
    main()

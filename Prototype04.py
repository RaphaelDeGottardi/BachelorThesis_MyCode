from itertools import combinations

import os
import numpy as np
import networkx as nx
import shutil


from graph_pkg_core.algorithm.graph_edit_distance import GED
from graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from graph_pkg_core.graph.edge import Edge
from graph_pkg_core.graph.graph import Graph
from graph_pkg_core.graph.label.label_edge import LabelEdge
from graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
from graph_pkg_core.graph.node import Node
from graph_pkg_core.loader.loader_vector import LoaderVector




def main():
    #shutil.rmtree(r'./MyCode/WL_graphs')

    k = 4                       # Nr of iterations (WL-Algorithm)
    nr_of_graphs = 5           # from Enzymes dataset

    #Create folders for the calculated (hash-augmented) graphs
    try:
        os.mkdir(f'./MyCode/WL_graphs') 
    except OSError as e:
        True

    #Loading the vectors and calculating their hash-values
    for grX in range(nr_of_graphs):        

        graph = nx.read_graphml(f'./Enzymes/data/gr_{str(grX)}.graphml')
        graph_hashes_list = nx.weisfeiler_lehman_subgraph_hashes(graph,
                                                                iterations=k,
                                                               digest_size=8)
      
        for node_index in graph.nodes:
            graph.nodes[node_index]['hash'] = str(graph_hashes_list[node_index])
    
        nx.write_graphml(
            graph, 
            f'./MyCode/WL_graphs/gr_{str(grX)}.graphml')

    #Loading the Graphs

    graph_list = []

    loader_vector = LoaderVector(f'./MyCode/WL_graphs')    
    original_graphs = loader_vector.load()

    #calculate GED: first normal, then add the distance of the hashed graphs
    #Graph index of the ones to compare (if X=Y -> ged=0)
    X = 0
    Y = 1

    ged_hash = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k=2))

    cost = ged_hash.compute_edit_distance(original_graphs[X],original_graphs[Y], heuristic=True)

    print(f'ged with WL: {cost}') 
     

    shutil.rmtree(r'./MyCode/WL_augmented_graphs')

if __name__ == "__main__":
    main()

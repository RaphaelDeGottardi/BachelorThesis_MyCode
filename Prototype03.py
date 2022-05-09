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
from graph_pkg_core.coordinator.coordinator_vector_classifier import CoordinatorVectorClassifier



def main():
    k = 5                       # Nr of iterations (WL-Algorithm)
    nr_of_graphs = 10           # from Enzymes dataset

    #Create folders for the calculated (hash-augmented) graphs
    try:
        os.mkdir(f'./MyCode/WL_augmented_graphs') 
    except OSError as e:
        True
    try:
        os.mkdir(f'./MyCode/WL_augmented_graphs/original_graphs') 
    except OSError as e:
        print('something is wrong')

    #Loading the vectors and calculating their hash-values
    for grX in range(nr_of_graphs):        

        graph = nx.read_graphml(f'./Enzymes/data/gr_{str(grX)}.graphml')
        graph_hashes_list = nx.weisfeiler_lehman_subgraph_hashes(graph,
                                                                iterations=k,
                                                               digest_size=8)
        #store original graph in own folder
        nx.write_graphml(
                graph, 
                f'./MyCode/WL_augmented_graphs/original_graphs/gr_{str(grX)}.graphml')
    
        #store hashed graphs in separate folders
        for iteration_k in range(k): 
            try:
                os.mkdir(f'./MyCode/WL_augmented_graphs/iteration{str(iteration_k+1)}_graphs') 
            except OSError as e:
                True
        
            for node_index in graph.nodes:
                graph.nodes[node_index]['hash'] = str(graph_hashes_list[node_index][iteration_k])

            nx.write_graphml(
                graph, 
                f'./MyCode/WL_augmented_graphs/iteration{str(iteration_k+1)}_graphs/gr_{str(grX)}.graphml')
    
    #load the graphs
    coordinator_vector = CoordinatorVectorClassifier('proteins',
                                              (1., 1., 1., 1., 'euclidean'),
                                              './graph-matching-core/tests/test_data/proteins_test')  




    shutil.rmtree(r'./MyCode/WL_augmented_graphs')

if __name__ == "__main__":
    main()

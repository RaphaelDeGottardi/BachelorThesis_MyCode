from itertools import combinations
from time import time

import os
import numpy as np
import networkx as nx
import shutil

#prototype to test the kNN classification aughmented with the WL-hashes

from graph_pkg_core.coordinator.coordinator_vector_classifier import CoordinatorVectorClassifier
from graph_pkg_core.algorithm.knn import KNNClassifier
from graph_pkg_core.utils.functions.helper import calc_accuracy


def main():

    k = 10                       # Nr of iterations (WL-Algorithm)
    nr_of_graphs = 4337          

    #remember to re-include the train test and val list files when having deleted it
    delete_old_graphs = False
    if delete_old_graphs:
        shutil.rmtree(r'./MyCode/WL_graphs_mutag')

    try:
        os.mkdir(f'./MyCode/WL_graphs_mutag') 
    except OSError as e:
        True

    #Loading the vectors and calculating their hash-values        
    for grX in range(nr_of_graphs):  
        
        print(f'graph nr. {grX} has been created')

        graph = nx.read_graphml(f'./mutagenicity/1651/gr_{str(grX)}.graphml')
        graph_hashes_list = nx.weisfeiler_lehman_subgraph_hashes(graph,
                                                                iterations=k,
                                                                digest_size=8)
    
        for node_index in graph.nodes:
            vector_label = str(graph.nodes[node_index]['x'])
            graph_hashes_list[node_index].insert(0,vector_label)
            graph.nodes[node_index]['hashes'] = str(graph_hashes_list[node_index])
    
        nx.write_graphml(
            graph, 
            f'./MyCode/WL_graphs_mutag/gr_{str(grX)}.graphml')


if __name__ == "__main__":
    main()

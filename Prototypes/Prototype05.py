from itertools import combinations

import os
import numpy as np
import networkx as nx
import shutil

#this is the first time trying to use the coordinator

from graph_pkg_core.algorithm.graph_edit_distance import GED
from graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from graph_pkg_core.graph.edge import Edge
from graph_pkg_core.graph.graph import Graph
from graph_pkg_core.graph.label.label_edge import LabelEdge
from graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
from graph_pkg_core.graph.node import Node
from graph_pkg_core.loader.loader_vector import LoaderVector
from graph_pkg_core.coordinator.coordinator_vector import CoordinatorVector
from graph_pkg_core.coordinator.coordinator_vector_classifier import CoordinatorVectorClassifier




def main():
    #shutil.rmtree(r'./MyCode/WL_graphs')

    k = 2                       # Nr of iterations (WL-Algorithm)
    nr_of_graphs = 600           # from Enzymes dataset max 

    #Create folders for the calculated (hash-augmented) graphs
    try:
        os.mkdir(f'./MyCode/WL_graphs') 
    except OSError as e:
        True

    #Loading the vectors and calculating their hash-values
    compute_hashed_graphs = False

    if compute_hashed_graphs==True:
        for grX in range(nr_of_graphs):        

            graph = nx.read_graphml(f'./Enzymes/data/gr_{str(grX)}.graphml')
            graph_hashes_list = nx.weisfeiler_lehman_subgraph_hashes(graph,
                                                                    iterations=k,
                                                                    digest_size=8)
        
            for node_index in graph.nodes:
                vector_label = str(graph.nodes[node_index]['x'])
                graph_hashes_list[node_index].insert(0,vector_label)
                graph.nodes[node_index]['hashes'] = str(graph_hashes_list[node_index])
        
            nx.write_graphml(
                graph, 
                f'./MyCode/WL_graphs/gr_{str(grX)}.graphml')


    #Loading the Graphs

    FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           'WL_graphs')
    #coordinator = CoordinatorVector('enzymes', (1., 1., 1., 1., 'euclidean', k),FOLDER_DATA,True) 
    classifier = CoordinatorVectorClassifier('enzymes',
                                    (1., 1., 1., 1., 'euclidean', k),
                                    FOLDER_DATA,None,True)

    
    X_train, y_train = getattr(classifier, 'train_split')()
    X_test, y_test = getattr(classifier, 'test_split')()
    X_validation, y_validation = getattr(classifier, 'val_split')()
    print()

if __name__ == "__main__":
    main()

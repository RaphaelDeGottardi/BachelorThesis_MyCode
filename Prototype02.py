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
    
    #load the graphs from the folder
    original_graphs = []
    iter_graphs = []

    loader_vector = LoaderVector(f'./MyCode/WL_augmented_graphs/original_graphs')    
    original_graphs = loader_vector.load()
    for iter_k in range(k):
        loader_vector_iter = LoaderVector(f'./MyCode/WL_augmented_graphs/iteration{str(iter_k+1)}_graphs', use_wl_attr=True)
        iter_graphs.append(loader_vector_iter.load())


    #calculate GED: first normal, then add the distance of the hashed graphs
    #Graph index of the ones to compare (if X=Y -> ged=0)
    X = 5
    Y = 6

    ged = GED(EditCostVector(1., 1., 1., 1., 'euclidean', use_wl_attr=False))  # EditCostVector takes weights (of cost function) as an input
    ged_hash = GED(EditCostVector(1., 1., 1., 1., 'euclidean', use_wl_attr=True))
    alpha = 0.1/k  #weight for how much the ged shoud be aughmented (by hash values)

    cost = ged.compute_edit_distance(original_graphs[X],original_graphs[Y], heuristic=True)
    print(f'ged before WL: {cost}') 

    for iter_k in range(k):
        cost += alpha*ged_hash.compute_edit_distance(iter_graphs[iter_k][X],iter_graphs[iter_k][Y], heuristic=True)
      
    print(f'ged with WL: {cost}')

    shutil.rmtree(r'./MyCode/WL_augmented_graphs')

if __name__ == "__main__":
    main()

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


#For now this code shows the use of the modified GED and that for k=0 it gives the same results

def main():
    shutil.rmtree(r'./MyCode/WL_graphs_exp')

    use_original_ged = True
    k = 0          #do not modify            
    if not use_original_ged:
        k = 2                   # Nr of iterations (WL-Algorithm)
    nr_of_graphs = 6            # from Enzymes dataset max 
    
    #Create folders for the calculated (hash-augmented) graphs
    try:
        os.mkdir(f'./MyCode/WL_graphs_exp') 
    except OSError as e:
        True

    #Loading the vectors and calculating their hash-values
    for grX in range(nr_of_graphs):        
        if use_original_ged: 
            graph = nx.read_graphml(f'./Enzymes/data/gr_{str(grX)}.graphml')
            nx.write_graphml(graph,  f'./MyCode/WL_graphs_exp/gr_{str(grX)}.graphml')
        else:
            graph = nx.read_graphml(f'./Enzymes/data/gr_{str(grX)}.graphml')
            graph_hashes_list = nx.weisfeiler_lehman_subgraph_hashes(graph,
                                                                    iterations=k,
                                                                    digest_size=8)
        
            for node_index in graph.nodes:
                vector_label = str(graph.nodes[node_index]['x'])
                if(k>0): 
                    graph_hashes_list[node_index].insert(0,vector_label)
                    graph.nodes[node_index]['hashes'] = str(graph_hashes_list[node_index])
                else:
                    graph.nodes[node_index]['hashes'] = vector_label
        
            nx.write_graphml(
                graph, 
                f'./MyCode/WL_graphs_exp/gr_{str(grX)}.graphml')


    #Loading the Graphs

    graph_list = []

    loader_vector = LoaderVector(f'./MyCode/WL_graphs_exp', use_wl_attr = not use_original_ged)    
    WL_graph_list = loader_vector.load()

    #calculate GED: first normal, then add the distance of the hashed graphs
    #Graph index of the ones to compare (if X=Y -> ged=0)
    X = 0
    Y = 1
    if use_original_ged:
        ged_WL = GED(EditCostVector(1., 1., 1., 1., 'euclidean'))
    else:
        ged_WL = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k = k))

    cost = ged_WL.compute_edit_distance(WL_graph_list[X],WL_graph_list[Y], heuristic=True)
    print(f'ged with WL: {cost}') 
     

if __name__ == "__main__":
    main()

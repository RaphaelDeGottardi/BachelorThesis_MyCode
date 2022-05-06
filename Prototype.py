from itertools import combinations

import os
import numpy as np
import networkx as nx


from graph_pkg_core.algorithm.graph_edit_distance import GED
from graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from graph_pkg_core.graph.edge import Edge
from graph_pkg_core.graph.graph import Graph
from graph_pkg_core.graph.label.label_edge import LabelEdge
from graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
from graph_pkg_core.graph.node import Node
from graph_pkg_core.loader.loader_vector import LoaderVector




def main():
    k = 2                       # Nr of iterations (WL-Algorithm)

    #Loading the Vectors and calculation their Hash-Values
    for grX in range(2):        # grX: nr of graphs to be read in (from Enzymes Dataset)

        graph = nx.read_graphml(f'./Enzymes/data/gr_{str(grX)}.graphml')
        graph_hashes_list = nx.weisfeiler_lehman_subgraph_hashes(graph,
                                                                iterations=k,
                                                               digest_size=8)
        #create folders to stor the Calculated graphs in    
        try:
            os.mkdir(f'./MyCode/HashedGraphs_P/gr_{str(grX)}')   
            os.mkdir(f'./MyCode/HashedGraphs_P/gr_{str(grX)}/hash')    
        except OSError as e:
            print('folder already exists')  
        
        #store original graph
        nx.write_graphml(
                graph, 
                f'./MyCode/HashedGraphs_P/gr_{str(grX)}/gr{str(grX)}.graphml')
    
        
        #store hashed graphs in subfolder
        for iteration_k in range(k): 
    
            for node_index in graph.nodes:
                graph.nodes[node_index]['hash'] = str(graph_hashes_list[node_index][iteration_k])

            graph.graph['WL_iteration'] = iteration_k + 1
            nx.write_graphml(
                graph, 
                f'./MyCode/HashedGraphs_P/gr_{str(grX)}/hash/gr{str(grX)}_hashed_{str(iteration_k+1)}.graphml')
    
    #calculate GED: first normal, then add the distance of the hashed graphs

    ged = GED(EditCostVector(1., 1., 1., 1., 'euclidean', use_wl_attr=False))  # EditCostVector takes weights (of cost function) as an input
    ged_hash = GED(EditCostVector(1., 1., 1., 1., 'euclidean', use_wl_attr=True))
    
    X = 0
    Y = 1
    #Doesn't work yet!
    loader_vectorX = LoaderVector(f'./MyCode/HashedGraphs_P/gr_{str(X)}', use_wl_attr=False)
    loader_vectorY = LoaderVector(f'./MyCode/HashedGraphs_P/gr_{str(Y)}')
    loader_vectorX_hash = LoaderVector(f'./MyCode/HashedGraphs_P/gr_{str(X)}/hash', use_wl_attr=True)
    loader_vectorY_hash = LoaderVector(f'./MyCode/HashedGraphs_P/gr_{str(Y)}/hash', use_wl_attr=True)
    
    graphX = loader_vectorX.load()
    graphY = loader_vectorY.load()   

    cost = ged.compute_edit_distance(graphX[0],graphY[0], heuristic=True)
    print(f'ged before WL: {cost}')

    graphX = loader_vectorX_hash.load()
    graphY = loader_vectorY_hash.load()   

    alpha = 0.1  #weight for how much the ged shoud be aughmented (by hash values)
    for iteration_k in range(k):
        cost += alpha*ged_hash.compute_edit_distance(graphX[iteration_k],graphY[iteration_k], heuristic=True)
        
    print(f'ged with WL: {cost}')

if __name__ == "__main__":
    main()

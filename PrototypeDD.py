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
    k = 2                   # Nr of iterations (WL-Algorithm)
    nr_of_graphs = 6            # from Enzymes dataset max 

    #Loading the Graphs

    graph_list = []

    loader_vector = LoaderVector(f'./MyCode/WL_graphs_dd', use_wl_attr = True)    
    WL_graph_list = loader_vector.load()

    #calculate GED: first normal, then add the distance of the hashed graphs
    #Graph index of the ones to compare (if X=Y -> ged=0)
    X = 0
    Y = 1

    ged_WL = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k = k, weights = [1/2,1/3]))

    cost = ged_WL.compute_edit_distance(WL_graph_list[X],WL_graph_list[Y], heuristic=True)
    print(f'ged with WL: {cost}') 
     

if __name__ == "__main__":
    main()

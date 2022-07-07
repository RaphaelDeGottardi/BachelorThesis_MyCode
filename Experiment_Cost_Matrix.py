from itertools import combinations
from time import time

import os
import numpy as np
import networkx as nx
import shutil
import itertools

#prototype to test the kNN classification aughmented with the WL-hashes

from graph_pkg_core.coordinator.coordinator_vector_classifier import CoordinatorVectorClassifier
from graph_pkg_core.algorithm.knn import KNNClassifier
from graph_pkg_core.utils.functions.helper import calc_accuracy
from graph_pkg_core.algorithm.graph_edit_distance import GED
from graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from graph_pkg_core.loader.loader_vector import LoaderVector


def main():

    k = 2                   # Nr of iterations (WL-Algorithm)
    nr_of_graphs = 6            # from Enzymes dataset max 

    #Loading the Graphs

    graph_list = []

    loader_vector = LoaderVector(f'./MyCode/WL_graphs_mutag', use_wl_attr = True)    
    WL_graph_list = loader_vector.load()


    X = 0
    Y = 1

    ged_WL = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k = k, weights = [1/2,1/3]))

    cost = ged_WL.compute_edit_distance(WL_graph_list[X],WL_graph_list[Y], heuristic=True)

    a = ged_WL.C
    print(a.base)

    print(f'ged with WL: {cost}') 


if __name__ == "__main__":
    main()

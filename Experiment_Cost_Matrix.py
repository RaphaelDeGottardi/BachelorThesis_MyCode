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


    loader_vector = LoaderVector(f'./MyCode/WL_graphs_dd', use_wl_attr = True)    
    WL_graph_list = loader_vector.load()


    X = 0
    Y = 1

    ged_WL = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k = 0))

    cost = ged_WL.compute_edit_distance(WL_graph_list[X],WL_graph_list[Y], heuristic=True)

    C_orig = ged_WL.C.base

    ged_WL = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k = 2, weights = [1/2,1/3]))

    cost = ged_WL.compute_edit_distance(WL_graph_list[X],WL_graph_list[Y], heuristic=True)


    C_wl = ged_WL.C.base

    C_diff = C_wl - C_orig

    np.savetxt("C_matrix_orig.csv",C_orig , 
              delimiter = ",")
    np.savetxt("C_matrix_wl.csv",C_wl , 
              delimiter = ",")
    np.savetxt("C_matrix_diff.csv",C_diff , 
              delimiter = ",")
    print(f'ged with WL: {cost}') 


if __name__ == "__main__":
    main()

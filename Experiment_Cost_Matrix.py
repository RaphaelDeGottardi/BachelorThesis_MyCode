from itertools import combinations
from time import time

import os
import numpy as np
import networkx as nx
import random as rd
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


    loader_vector = LoaderVector(f'./MyCode/WL_graphs_enzymes', use_wl_attr = True)    
    WL_graph_list = loader_vector.load()




    ged_orig = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k = 0))
    ged_WL = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k = 2, weights = [1/2,1/3]))

    # for i in range(20):
    X = 11 #rd.randint(0,len(WL_graph_list))
    Y = 56 #rd.randint(0,len(WL_graph_list))
    print(f'the graphs X: {X} and Y: {Y} were chosen from the enzymes dataset')

    cost_orig = ged_orig.compute_edit_distance(WL_graph_list[X],WL_graph_list[Y], heuristic=True)
    C_orig = ged_orig.C.base

    Zcount_orig = 0
    for line in C_orig:
        for element in line:
            if element == 0:
                Zcount_orig = Zcount_orig + 1
    print(f' Zeroes in original C_matrix: {Zcount_orig}')


    cost = ged_WL.compute_edit_distance(WL_graph_list[X],WL_graph_list[Y], heuristic=True)

    C_wl = ged_WL.C.base

    Zcount_wl = 0

    for line in C_wl:
        for element in line:
            if element == 0:
                Zcount_wl = Zcount_wl + 1
    print(f' Zeroes in wl C_matrix: {Zcount_wl}')

    print(f'the difference in nr of zeroes is: {Zcount_orig-Zcount_wl}')

    C_diff = C_wl - C_orig

    np.savetxt("C_matrix_orig_toyexp.csv",C_orig , 
              delimiter = ",")
    np.savetxt("C_matrix_wl_toyexp.csv",C_wl , 
              delimiter = ",")
    np.savetxt("C_matrix_diff_toyexp.csv",C_diff , 
              delimiter = ",")
    print(f'ged with WL: {cost}') 


if __name__ == "__main__":
    main()

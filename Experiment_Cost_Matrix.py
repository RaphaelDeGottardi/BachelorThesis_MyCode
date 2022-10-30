from itertools import combinations
from time import time


# importing the required modules
import matplotlib.pyplot as plt
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


    loader_vector = LoaderVector(f'./MyCode/WL_graphs_enzymes/1651', use_wl_attr = True)    
    WL_graph_list = loader_vector.load()


    print('this experiment documents the zero reduction for enzymes and a ged with weights (1/2,0,0)')

    ged_orig = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k = 0))
    ged_WL = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k = 2, weights = [1/2,0]))
    ged_WL2 = GED(EditCostVector(1., 1., 1., 1., 'euclidean', wl_k = 2, weights = [1/2,1/2]))

    zerocount = []

    for i in range(1):
        X = rd.randint(0,len(WL_graph_list)-1)
        Y = rd.randint(0,len(WL_graph_list)-1)
        print(f'the graphs X: {X} and Y: {Y} were chosen from the enzymes dataset')

        cost_orig = ged_orig.compute_edit_distance(WL_graph_list[X],WL_graph_list[Y], heuristic=True)
        C_orig = ged_orig.C.base

        Zcount_orig = 0
        for y in range(len(WL_graph_list[Y].nodes)):
            for x in range(len(WL_graph_list[X].nodes)):
                if C_orig[x][y] == 0:
                    Zcount_orig = Zcount_orig + 1


        cost = ged_WL.compute_edit_distance(WL_graph_list[X],WL_graph_list[Y], heuristic=True)

        C_wl = ged_WL.C.base
        Zcount_wl = 0

        for y in range(len(WL_graph_list[Y].nodes)):
            for x in range(len(WL_graph_list[X].nodes)):
                if C_wl[x][y] == 0:
                    Zcount_wl = Zcount_wl + 1

        zerocount.append(Zcount_wl/Zcount_orig)

        
        cost = ged_WL2.compute_edit_distance(WL_graph_list[X],WL_graph_list[Y], heuristic=True)

        C_wl2 = ged_WL2.C.base
    print('this is the vector including the nr of zeros:')
    print(zerocount)
    print(f'mean:{np.mean(zerocount)}')
    print(f'stabw:{np.std(zerocount)}')
    

    # np.savetxt("C_matrix_orig_toyexp.csv",C_orig , 
    #           delimiter = ",")
    # np.savetxt("C_matrix_wl_toyexp.csv",C_wl , 
    #           delimiter = ",")

    plt.matshow(C_orig, cmap='Greys')
    plt.title('Original')
    plt.colorbar()
    plt.matshow(C_wl, cmap='Greys')
    plt.title('o + h(1)')
    plt.colorbar()    
    plt.matshow(C_wl2, cmap='Greys')
    plt.title('o + h(1) + h(2)')
    plt.colorbar()
    plt.show()
    print(f'ged with WL: {cost}') 


if __name__ == "__main__":
    main()

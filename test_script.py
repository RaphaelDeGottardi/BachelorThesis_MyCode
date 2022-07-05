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


def main():

    #Loading the Graphs
    print("milestone1")
    FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           'WL_graphs_enzymes')

    wl_k = 3 #for WL algorithm

    weights = [] #list(itertools.product([0,0.1,0.2,0.3], repeat=3))
    add_weights = [[0,0,0.2],[1/4,1/8,1/16],[1/8,1/16,1/32],[1/2,1/4,1/4],[2/3,1/6,1/6],[1/3,1/3,1/3],[1/3,1/3,1/31]]
    for weight in add_weights:
        weights.append(weight)
    print("created weights")


    for weight in weights:
        print("milestone2")

        
if __name__ == "__main__":
    main()

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

    FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           'WL_graphs_enzymes')

    wl_k = 3 #for WL algorithm

    weights = list(itertools.product([0,0.1,0.2,0.3], repeat=3))
    add_weights = [[1/8,0,0],[1/4,1/8,1/16],[1/8,1/16,1/32],[1/2,1/4,1/4],[2/3,1/6,1/6],[1/3,1/3,1/3],[1/3,1/3,1/31]]
    for weight in add_weights:
        weights.append(weight)

    for weight in weights:
        coordinator = CoordinatorVectorClassifier('enzymes',
                                        (1., 1., 1., 1., 'euclidean',wl_k,[weight[0],weight[1],weight[2]]),
                                        FOLDER_DATA,None,True,False)

    
        X_train, y_train = getattr(coordinator, 'train_split')()
        #X_test, y_test = getattr(coordinator, 'test_split')()
        X_validation, y_validation = getattr(coordinator, 'val_split')()
        
        knn = KNNClassifier(coordinator.ged, True, verbose=False)
        knn.train(graphs_train=X_train, labels_train=y_train)

        start_time = time()
        predictions = knn.predict(X_validation, k=1, num_cores=8)
        prediction_time = time() - start_time
        acc = calc_accuracy(np.array(y_validation, dtype=np.int32),
                            np.array(predictions, dtype=np.int32))

        message = f'Weights and best acc on val : {weight}:{acc:.2f}, time: {prediction_time:.2f}s\n'

        print(message)

if __name__ == "__main__":
    main()

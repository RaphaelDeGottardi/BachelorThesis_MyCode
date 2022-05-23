from itertools import combinations
from time import time

import os
import numpy as np
import networkx as nx
import shutil

#prototype to test the kNN classification aughmented with the WL-hashes

from graph_pkg_core.coordinator.coordinator_vector_classifier import CoordinatorVectorClassifier
from graph_pkg_core.algorithm.knn import KNNClassifier
from graph_pkg_core.utils.functions.helper import calc_accuracy


def main():

    wl_k = 0                       # Nr of iterations (WL-Algorithm)

    #Loading the Graphs

    FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           'WL_graphs')

    coordinator = CoordinatorVectorClassifier('enzymes',
                                    (1., 1., 1., 1., 'euclidean',wl_k),
                                    FOLDER_DATA,None,True,False)

   
    X_train, y_train = getattr(coordinator, 'train_split')()
    X_test, y_test = getattr(coordinator, 'test_split')()
    X_validation, y_validation = getattr(coordinator, 'val_split')()
    
    knn = KNNClassifier(coordinator.ged, True, verbose=False)
    knn.train(graphs_train=X_train, labels_train=y_train)

    start_time = time()
    predictions = knn.predict(X_test, k=3, num_cores=6)
    prediction_time = time() - start_time
    acc = calc_accuracy(np.array(y_test, dtype=np.int32),
                        np.array(predictions, dtype=np.int32))

    message = f'Best acc on Test : {acc:.2f}, time: {prediction_time:.2f}s\n'

    print(message)

if __name__ == "__main__":
    main()

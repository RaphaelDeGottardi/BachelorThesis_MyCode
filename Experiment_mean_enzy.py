from itertools import combinations
from time import time

import os
import numpy as np
import networkx as nx
import shutil
import itertools

#Experiment to calculate the mean accuracy of the prediction on the original graphs

from graph_pkg_core.coordinator.coordinator_vector_classifier import CoordinatorVectorClassifier
from graph_pkg_core.algorithm.knn import KNNClassifier
from graph_pkg_core.utils.functions.helper import calc_accuracy


def main():
   #Loading the Graphs
    shufflelist = [1651,21465,27709,44332,53628,54292,56261,60278,77558,80822]
    for kNN in [1,3]:

        for shuffle in shufflelist:
            print(f'Experiment to calculate the mean accuracy(k={kNN}) of the prediction on the original graphs, seed:{shuffle}')
    
            shuffle_str = str(shuffle)
            FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                                f'WL_graphs_enzymes/{shuffle}')

            wl_k = 3 #for WL algorithm


            coordinator = CoordinatorVectorClassifier('enzymes',
                                            (1., 1., 1., 1., 'euclidean',wl_k,[0,0,0]),
                                            FOLDER_DATA,None,True,False)

            X_train, y_train = getattr(coordinator, 'train_split')()
            X_test, y_test = getattr(coordinator, 'test_split')()
            X_validation, y_validation = getattr(coordinator, 'val_split')()

            knn = KNNClassifier(coordinator.ged, True, verbose=False)
            knn.train(graphs_train=X_train, labels_train=y_train)

            start_time = time()
            predictions = knn.predict(X_validation, k=1, num_cores=8)
            prediction_time = time() - start_time
            acc = calc_accuracy(np.array(y_validation, dtype=np.int32),
                                np.array(predictions, dtype=np.int32))    
            message = f'Best acc on val: {acc:.2f}, time: {prediction_time:.2f}s\n'
            print(message)  

            start_time = time()
            predictions = knn.predict(X_test, k=1, num_cores=8)
            prediction_time = time() - start_time
            acc = calc_accuracy(np.array(y_test, dtype=np.int32),
                                np.array(predictions, dtype=np.int32))

            message = 'Best acc on test: {acc:.2f}, time: {prediction_time:.2f}s\n'
            print(message)

if __name__ == "__main__":
    main()

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

    k = 1                       # Nr of iterations (WL-Algorithm)
    nr_of_graphs = 600           # from Enzymes dataset max 

    #Loading the vectors and calculating their hash-values
    #remember to re-include the train test and val list files when having deleted it
    compute_hashed_graphs = False
    if compute_hashed_graphs:
        shutil.rmtree(r'./MyCode/WL_graphs')
            #Create folders for the calculated (hash-augmented) graphs ( after deleting the old ones)
        try:
            os.mkdir(f'./MyCode/WL_graphs') 
        except OSError as e:
            True
        for grX in range(nr_of_graphs):        

            graph = nx.read_graphml(f'./Enzymes/data/gr_{str(grX)}.graphml')
            graph_hashes_list = nx.weisfeiler_lehman_subgraph_hashes(graph,
                                                                    iterations=k,
                                                                    digest_size=8)
        
            for node_index in graph.nodes:
                vector_label = str(graph.nodes[node_index]['x'])
                graph_hashes_list[node_index].insert(0,vector_label)
                graph.nodes[node_index]['hashes'] = str(graph_hashes_list[node_index])
        
            nx.write_graphml(
                graph, 
                f'./MyCode/WL_graphs/gr_{str(grX)}.graphml')


    #Loading the Graphs

    FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           'WL_graphs')

    coordinator = CoordinatorVectorClassifier('enzymes',
                                    (1., 1., 1., 1., 'euclidean',k),
                                    FOLDER_DATA,None,True,False)

   
    X_train, y_train = getattr(coordinator, 'train_split')()
    X_test, y_test = getattr(coordinator, 'test_split')()
    #X_validation, y_validation = getattr(coordinator, 'val_split')()
    
    #knn
    #use train() on X_train
    #use predict() in X_val
    #compare using utils.functions.helper import calc_accuracy
    #link to knn github: https://github.com/CheshireCat12/graph-matching-gnn-reduction/blob/master/knn/run_knn.pyx

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

from itertools import combinations

import os
import numpy as np
import networkx as nx
import shutil



from graph_pkg_core.algorithm.graph_edit_distance import GED
from graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from graph_pkg_core.coordinator.coordinator_vector_classifier import CoordinatorVectorClassifier
from graph_pkg_core.algorithm.matrix_distances import MatrixDistances


def main():

    k = 1                        # Nr of iterations (WL-Algorithm)
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
    #coordinator = CoordinatorVector('enzymes', (1., 1., 1., 1., 'euclidean', k),FOLDER_DATA,True) 
    classifier = CoordinatorVectorClassifier('enzymes',
                                    (1., 1., 1., 1., 'euclidean', k, 0.8),
                                    FOLDER_DATA,None,True,False)
   
    X_train, y_train = getattr(classifier, 'train_split')()

    #run it in paralell ( faster), num_cores = 6 ( up to 8)
    #use the MatrixDistances Class and the calc_matr_dist (X_train, X_train)

    matrix_dist = MatrixDistances(classifier.ged,
                               parallel = True)
    dist = matrix_dist.calc_matrix_distances(X_train, X_train, heuristic=True, num_cores=6)


    X_test, y_test = getattr(classifier, 'test_split')()
    X_validation, y_validation = getattr(classifier, 'val_split')()
    print()


    #knn
    #use train() on X_train
    #use predict() in X_val
    #compare using utils.functions.helper import calc_accuracy

if __name__ == "__main__":
    main()

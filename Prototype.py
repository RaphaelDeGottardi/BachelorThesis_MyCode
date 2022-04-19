from itertools import combinations

import numpy as np
import networkx as nx


from graph_pkg_core.algorithm.graph_edit_distance import GED
from graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from graph_pkg_core.graph.edge import Edge
from graph_pkg_core.graph.graph import Graph
from graph_pkg_core.graph.label.label_edge import LabelEdge
from graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
from graph_pkg_core.graph.node import Node




def main():
    k = 2                       # Nr of iterations (WL-Algorithm)
    for grX in range(3):        # grX: nr of graphs to be read in (from Enzymes Dataset)

        graph = nx.read_graphml('/home/raphaeldegottardi/BachelorThesis/Enzymes/data/gr_'+ str(grX)+ '.graphml')
        graph_hashes_list = nx.weisfeiler_lehman_subgraph_hashes(graph, iterations=k, digest_size=8)

        for iteration_k in range(k): 
    
            for node_index in graph.nodes:
                graph.nodes[node_index]['hash'] = str(graph_hashes_list[node_index][iteration_k])

            graph.graph['WL_iteration'] = iteration_k+1
            nx.write_graphml( graph , '/home/raphaeldegottardi/BachelorThesis/MyCode/HashedGraphs_P/gr'+str(grX)+'_hashed_'+str(iteration_k+1)+'.graphml')


if __name__ == "__main__":
    main()

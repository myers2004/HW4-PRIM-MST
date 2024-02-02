import numpy as np
import heapq as hq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    
    
    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        self.mst = None

        #first we have to make our pq updatable

        pq = []                         # list of entries arranged in a heap
        entry_finder = {}               # mapping of tasks to entries
        REMOVED = '<removed-task>'      # placeholder for a removed task

        def add_task(task, priority=0, count=0):
            'Add a new task or update the priority of an existing task'
            if task in entry_finder:
                remove_task(task)
            entry = [priority, count, task]
            entry_finder[task] = entry
            hq.heappush(pq, entry)

        def remove_task(task):
            'Mark an existing task as REMOVED.  Raise KeyError if not found.'
            entry = entry_finder.pop(task)
            entry[-1] = REMOVED

        def pop_task():
            'Remove and return the lowest priority task. Raise KeyError if empty.'
            while pq:
                priority, count, task = hq.heappop(pq)
                if task is not REMOVED:
                    del entry_finder[task]
                    return task
            raise KeyError('pop from an empty priority queue')

        S = []
        T = {}

        num_rows, num_col = np.shape(self.adj_mat)

        last_node = {i: None for i in range(num_rows)}

        s = 0
        pi = {}

        pi[s] = 0

        inf = 999999999


        for v in range(num_rows):
            if v != s:
                pi[v] = inf
        for v in range(num_rows):
            add_task(v, pi[v], v)
        i = 0
        while len(T) < num_rows - 1:
            u = pop_task()
            S.append(u)
            if last_node[u] != None:
                T[u] = last_node[u]
            for v in range(num_col):
                if self.adj_mat[u][v] != 0 and v not in S:
                    if self.adj_mat[u][v] < pi[v]:
                        pi[v] = self.adj_mat[u][v]
                        add_task(v, pi[v],i)
                        last_node[v] = u
                        i += 1
        
        mst = np.zeros((num_rows,num_col))
        for key in T.keys():
            mst[key][T[key]] = 1
            mst[T[key]][key] = 1
        self.mst = mst
        return(mst)




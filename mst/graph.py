import numpy as np
import heapq as hq
from typing import Union
import os

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            if os.stat(adjacency_mat).st_size == 0:
                raise ImportError('Input file is empty')
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        
        num_nodes, num_col = np.shape(self.adj_mat)
        row_sums = np.sum(self.adj_mat, axis=1)
        for i in range(num_nodes):
            if row_sums[i] < .00001:
                raise ValueError('Input matrix had unconnected node')
        if num_nodes != num_col:
            raise ValueError('Input matrix is has asymmetric dimensions')
        
        
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
        #Thankfully, the heapq documentation has exactly how to do that:
        # The three functions below are from https://docs.python.org/3/library/heapq.html
        # with a slight modification to the return in pop_task()

        #Uses a dictionary to make entires indexable and if an entyr is updated, sets the
        # name of the original entry to <removed-task>

        pq = []                         # list of entries arranged in a heap, our pq
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
            'Remove and return the lowest priority task. Returns Empty Queue if empty'
            'We will use this return to stop Prims algorithm'
            while pq:
                priority, count, task = hq.heappop(pq)
                if task is not REMOVED:
                    del entry_finder[task]
                    return task
            return 'Empty Queue'

        S = []                      #To store the explored nodes
        T = {}                      #To store the edge index to include in our mst

        num_nodes = np.shape(self.adj_mat)[0]

        last_node = {i: None for i in range(num_nodes)} #Will hold each node's predecessor (pred)

        s = np.random.randint(0,num_nodes, 1)[0] #Get a random start node

        pi = {} #Will hold the edge weight (distance) of the edge connecting unexplored node to the expolored reigon

        pi[s] = 0 #set the distance to the start node to zero

        inf = 999999999

        #Start Prim's alogorithm

        #Intilize all nodes but start with infinite distance to explored reiogn
        for v in range(num_nodes):
            if v != s:
                pi[v] = inf
        
        #Add all nodes with their priority and step added to pq
        for v in range(num_nodes):
            add_task(v, pi[v], v)

        #Go through the pq, adding the closest node to the explored reigon connected
        # to the explored reigon and the connecting edge to the mst. Update priority of 
        # nodes in unexplored reigon
        i = 0                                            #stores step count a node is added to pq, as a tie breaker
        while len(pq) > 0:
            u = pop_task()                               #get lowest priorty node, call it u
            if u == 'Empty Queue':                       #if pop_task reports the queue is empty, stop
                break
            S.append(u)                                  #add u to the explored region
            
            if last_node[u] != None:                     #for every node but the start node, add its pred to T
                T[u] = last_node[u]
            
            #If an edge exists connecting u and an unexplored node v, AND the edge weight 
            # connecting u to v is less than the current distance from the explored reigon
            # to v, update v's priority(distance) with that edge weight and its pred with u
            for v in range(num_nodes):
                if self.adj_mat[u][v] != 0 and v not in S:
                    if self.adj_mat[u][v] < pi[v]:
                        pi[v] = self.adj_mat[u][v]        #update distance
                        add_task(v, pi[v],i)              #update priority
                        last_node[v] = u                  #update pred
                        i += 1                            #add 1 to step count

        
        #Go from dict T to np.array mst
        mst = np.zeros((num_nodes,num_nodes))              #intilize mst
        #If an edge index is in our mst dict representation, add its weight to our mst array representation
        for key in T.keys():
            mst[key][T[key]] = self.adj_mat[key][T[key]]   
            mst[T[key]][key] = self.adj_mat[T[key]][key]        #our mst is symmetric
        self.mst = mst                                          #store our mst
        return mst




import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    #Check mst weight for against known correct weight
    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    #Check that mst has number of nodes - 1 edges
    num_nodes = np.shape(adj_mat)[0]
    num_mst_edges = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            if mst[i][j] > 0:
                num_mst_edges += 1
    assert num_mst_edges == num_nodes - 1,       'Proposed MST has wrong number of edges'

    #Check that mst is symmetric
    for i in range(num_nodes):
        for j in range(num_nodes):
            assert approx_equal(mst[i][j], mst [j][i]),      'Proposed MST is asymmetric'


    #Check that every node is visited in the mst
    row_sums = np.sum(mst, axis=1)
    for i in range(num_nodes):
        assert row_sums[i] > 0,                              'A node is not connected'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    #Check an error is raised if an m x n matrix where m != n is input
    file_path = './data/asymmetric.csv'
    with pytest.raises(ValueError):
        asym = Graph(file_path)

    #Check an error is raised if an empty file is input
    file_path = './data/empty.csv'
    with pytest.raises(ImportError):
        empty = Graph(file_path)

    #Check an error is raised if an adj matrix with a node with no edges
    file_path = './data/unconnected.csv'
    with pytest.raises(ValueError):
        unconnected = Graph(file_path)

    #Check an error is raised if a file is load in that doesn't exist
    file_path = './data/doesnt_exist.csv'
    with pytest.raises(FileNotFoundError):
        no_file = Graph(file_path)

    #Check an error is raised if type of argument is not string or np.array
    with pytest.raises(TypeError):
        wrong_type = Graph(32.4)


    


    
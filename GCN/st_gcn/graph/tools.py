import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    return np.dot(A, Dn)


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    return np.dot(np.dot(Dn, A), Dn)


def get_uniform_graph(num_node, self_link, neighbor):
    return normalize_digraph(edge2mat(neighbor + self_link, num_node))


def get_uniform_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    return I - N


def get_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    return np.stack((I, N))


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    return np.stack((I, In, Out))


def get_DAD_graph(num_node, self_link, neighbor):
    return normalize_undigraph(edge2mat(neighbor + self_link, num_node))


def get_DLD_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    return I - normalize_undigraph(edge2mat(neighbor, num_node))

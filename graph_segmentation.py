import numpy as np

k = 1
image = np.array((100,100))
width, height = image.shape[0:1]

class GraphSegmentator():
    '''Class that is capable of generating a graph based segmentation of the given image.
    Handles cluster objects that represent a set of vertices, and support union-find operations.
    '''
    def __init__(self, image, k):
        '''Constructor.
        ----------
        Parameters:
            image: 
                The image that the clustering is based upon
            k:
                The free parameter of the segmentation. 
                This determines how large each cluster 'wants' to be.
        '''
        self.image = image
        self.threshold = lambda cluster: k / len(cluster)

        def distance(vertex1, vertex2):
            return np.linalg.norm(self.image[vertex1], self.image[vertex2])

        # Nested dictionaries, where every pixel is a key in the outer dictionary
        # and only those pixel's neighbours are keys in the inner dictionary
        self.weights = {
            vertex: {
                neighbour: distance(vertex, neighbour) for neighbour in self.neighbours(vertex)
            } for vertex in image
        }

    def neighbours(self, vertex):
        '''Returns the set of 8-connected neighbouring vertices in the image.
        Can be less than 8 elements if node is in an edge.
        ----------
        Parameters:
            vertex:
                vertex in the image.
        '''
        x = vertex[0]
        y = vertex[1]

        neighbours = []
        if x - 1 >= 0:
            if y - 1 >= 0:
                neighbours.append(np.array([x - 1, y - 1]))
            neighbours.append(np.array([x - 1, y]))
            if y + 1 < height:
                neighbours.append(np.array([x - 1, y + 1]))
        if y - 1 >= 0:
            neighbours.append(np.array([x, y - 1]))
        if y + 1 < height:
            neighbours.append(np.array([x, y + 1]))
        if x + 1 < width:
            if y - 1 >= 0:
                neighbours.append(np.array([x + 1, y - 1]))
            neighbours.append(np.array([x + 1, y]))
            if y + 1 < height:
                neighbours.append(np.array([x + 1, y + 1]))
        return neighbours
 
    def cluster_edges(self, cluster):
        '''Returns the set of vertices that are neighbour to a vertex in
        the cluster, but isn't actually in the cluster.
        ----------
        Parameters:
            cluster1:
                iterable of vertices that supports union-find
        Returns:
            dictionary where the keys are vertices in the border and the
            values are their neighbours in the cluster
        '''
        border = {}
        for vertex in cluster:
            for neighbour in self.neighbours(vertex):
                if not cluster.find(neighbour):
                    if neighbour in border:
                        border[neighbour].append(vertex)
                    else:
                        border[neighbour] = [vertex]
        return border

    def mst(self, cluster):
        '''Returns the weights of the edges in the minimal spanning tree of 
        the given cluster. This is donde by a greedy algorithm, that chooses 
        the edges with the lowest weights to build the tree.
        ----------
        Parameters:
            image:
                image to use for vertex values
            cluster:
                iterable of vertices that supports union-find
        Returns:
            an iterable of weights (floats) that supports union-find
        '''
        pass

    def internal_dif(self, cluster):
        '''Returns the internal difference of a cluster
        ----------
        Parameters:
            image:
                image to use for vertex values
            cluster:
                iterable of vertices that supports union-find
        Returns:
            float
        '''
        return min(self.mst(cluster))

    def minimum_internal_dif(self, cluster1, cluster2):
        '''Returns the minimum internal difference between two clusters
        ----------
        Parameters:
            image:
                image to use for vertex values
            cluster1, cluster2:
                iterable of vertices that supports union-find
        Returns:
            float
        '''
        threshold1, threshold2 = k / len(cluster1), k / len(cluster2)
        return min(self.internal_dif(cluster1) + threshold1, self.internal_dif(cluster2) + threshold2)

    def dif(self, cluster1, cluster2):
        '''Returns the difference between two clusters
        ----------
        Parameters:
            image:
                image to use for vertex values
            cluster1, cluster2:
                iterable of vertices that supports union-find
        Returns:
            float
        '''
        border = self.cluster_edges(cluster1)
        border_weights = []
        for vertex in border.keys():
            # If vertex is in cluster2
            if cluster2.find(vertex):
                # For every vertex in cluster1 that has the vertex in cluster2 as a neighbour
                for cluster1vertex in border[vertex]:
                    border_weights.append(self.weights[vertex][cluster1vertex])

    def disjointed(self, cluster1, cluster2):
        '''Evaluates the predicate proposed in the paper to check if there 
        exists evidence that two clusters have a boundary between them.
        ----------
        Parameters:
            image:
                image to use for vertex values
            cluster1, cluster2:
                iterable of vertices that supports union-find
        Returns:
            True if the is evidence of a boundary, false otherwise
        '''
        return self.dif(cluster1, cluster2) > self.minimum_internal_dif(cluster1, cluster2)

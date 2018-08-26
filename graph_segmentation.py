import numpy as np
import re
import os
import time
from vertex import Vertex
from edge import Edge

class GraphSegmentator():
    '''Class that is capable of generating a graph based segmentation of the given image.
    Handles cluster objects that represent a set of vertices, and support union-find operations.
    Stores edge data in a temporary file, deleted when the segmentation is completed.
    '''
    def __init__(self, image, k, store_data=False):
        '''Constructor.
        ----------
        Parameters:
            image: 
                The image that the clustering is based upon
            k:
                The free parameter of the segmentation. 
                This determines how large each cluster 'wants' to be.
        '''
        self.store_data = store_data

        self.image = image
        self.width, self.height, _ = self.image.shape
        self.threshold = lambda vertex: k / len(self.clusters[self.vertices[vertex.coordinates].find().coordinates])

        def iterate_image():
            for x in range(self.width):
                for y in range(self.height):
                    yield (x, y)
        
        self.image_iterator = iterate_image

        # Matrix that contains the vertex object for each coordinates
        self.vertices = np.asanyarray([
            [
                Vertex((x, y)) for y in range(self.height)
            ] for x in range(self.width)
        ])

        self.clusters = {vertex: [vertex] for vertex in self.image_iterator()}

        # We must parse the edges right away to have efficient mst calculations later on.
        # Because of this, we start the algorithm right away, by merging two vertices if a weight is 0.
        # This saves up a lot of resources, because most of the edges in an image have weight 0.
        self.edges = {}

        c = 0
        for vertex in self.image_iterator():
            print("Calculating edges... {:.2%}".format(c / self.width / self.height), end="\r", flush=True)
            c += 1
            for neighbour in self.neighbours(vertex):
                dist = self.distance(vertex, neighbour)
                # Try to merge vertices if distance is 0.
                if dist == 0:
                    root1, root2 = self.vertices[vertex].find(), self.vertices[neighbour].find()
                    # Merge if on different clusters
                    if root1 != root2:
                        # Update storage structures
                        self.clusters[root1.coordinates].extend(self.clusters[root2.coordinates])
                        del self.clusters[root2.coordinates]
                        # Update union-find structure        
                        root1.unite(root2)
                # Skip edge if already stored the reflection
                if (neighbour, vertex) in self.edges:
                    continue
                # If not, store edge in dictionary
                self.edges[(vertex, neighbour)] = Edge(vertex, neighbour, distance = dist)
        print("Calculating edges... {:.2%}".format(c / self.width / self.height), end="\r", flush=True)

    def get_edge(self, vertex1, vertex2):
        '''Returns the weight of the edge that connects vertex1 and vertex2.
        Assumes given vertices are neighbourss in the image
        '''
        if (vertex1, vertex2) in self.edges:
            return self.edges[(vertex1, vertex2)]
        elif (vertex2, vertex1) in self.edges:
            return self.edges[(vertex2, vertex1)]
        else:
            return Edge(vertex1, vertex2, distance = 0)

    def distance(self, vertex1, vertex2):
        return np.linalg.norm(self.image[vertex1] - self.image[vertex2])

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
                neighbours.append((x - 1, y - 1))
            neighbours.append((x - 1, y))
            if y + 1 < self.height:
                neighbours.append((x - 1, y + 1))
        if y - 1 >= 0:
            neighbours.append((x, y - 1))
        if y + 1 < self.height:
            neighbours.append((x, y + 1))
        if x + 1 < self.width:
            if y - 1 >= 0:
                neighbours.append((x + 1, y - 1))
            neighbours.append((x + 1, y))
            if y + 1 < self.height:
                neighbours.append((x + 1, y + 1))
        return neighbours

    def mst(self, root):
        '''Returns the weights of the edges in the minimal spanning tree of 
        the given cluster. This is donde by a greedy algorithm, that chooses 
        the edges with the lowest weights to build the tree.
        ----------
        Parameters:
            image:
                image to use for vertex values
            root:
                vertex that is the root of a cluster
        Returns:
            an iterable of weights (floats) that supports union-find
        '''
        # Dictionary of union-find nodes for the algorithm
        mst = {}
        edges = []
        for vertex in self.clusters[root.coordinates]:
            for neighbour in self.neighbours(vertex):
                if self.vertices[vertex].find() == self.vertices[neighbour].find():
                    edges.append(self.get_edge(vertex, neighbour))
            mst[vertex] = Vertex(vertex)

        mst_weights = []

        for edge in sorted(edges, key=lambda edge: edge.weight):
            # Edge has already been counted
            if mst[edge.vertex1].find() == mst[edge.vertex2].find():
                continue
            # There is no edge connecting the nodes in the mst
            else:
                # Connect the nodes with this edge
                mst[edge.vertex1].find().unite(mst[edge.vertex2].find())
                mst_weights.append(edge.weight)

            if len(mst.keys()) == len(self.clusters[root.coordinates]):
                break

        return mst_weights if len(mst_weights) != 0 else [0]

    def internal_dif(self, root):
        '''Returns the internal difference of a cluster
        ----------
        Parameters:
            image:
                image to use for vertex values
            cluster:
                vertex that is a root
        Returns:
            float
        '''
        return max(self.mst(root))

    def minimum_internal_dif(self, root1, root2):
        '''Returns the minimum internal difference between two clusters
        ----------
        Parameters:
            image:
                image to use for vertex values
            root1, root2:
                vertices that are roots
        Returns:
            float
        '''
        return min(self.internal_dif(root1) + self.threshold(root1), self.internal_dif(root2) + self.threshold(root2))

    def segment(self):
        '''Applies the algorithm explained in the paper.
        '''
        # Algorithm
        remaining_edges = len(self.edges)
        c = 0
        for pair in sorted(self.edges, key = lambda elem: self.edges[elem].weight):
            print("Processing edges... {:.2%}   ".format(c / remaining_edges), end="\r", flush=True)
            c += 1
            weight, vertex1, vertex2 = self.edges[pair].weight, pair[0], pair[1]
            root1, root2 = self.vertices[vertex1].find(), self.vertices[vertex2].find()
            if (root1 != root2 and
                weight < self.minimum_internal_dif(root1, root2)):
                # Mutate the dictionary to merge the new clusters
                self.clusters[root1.coordinates].extend(self.clusters[root2.coordinates])
                del self.clusters[root2.coordinates]
                # Mutate the union-find to update with the join
                root1.unite(root2)
        print("Processing edges... {:.2%}".format(c / remaining_edges), end="\r", flush=True)

        return self.clusters.values()

if __name__ == '__main__':
    import skimage.io
    import skimage.color
    import datetime
    import colorsys

    def segment_image(img, result_name):
        start = datetime.datetime.now()
        print("Starting at {}.".format(start))
        g = GraphSegmentator(img, 1000, store_data=True)
        clusters = g.segment()
        finish = datetime.datetime.now()
        print("Finished at {}.".format(finish))
        print("Calculation time: {}".format(finish - start))

        N = len(clusters)
        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

        new_img = np.zeros(img.shape)

        for cluster, color in zip(clusters, RGB_tuples):
            for pixel in cluster:
                new_img[pixel] = color

        skimage.io.imsave(result_name, new_img)

    img = skimage.io.imread("images/image_6.png")

    img = skimage.color.convert_colorspace(img, 'RGB', 'HSV')

    segment_image(img, "result_image6HSV.jpg")

    img = skimage.io.imread("images/image_6.png")

    img = skimage.color.rgb2lab(img)

    segment_image(img, "result_image6LAB.jpg")
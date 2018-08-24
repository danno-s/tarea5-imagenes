import numpy as np
import re
import os
from vertex import Vertex

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

        self.edges_filename = 'edges.tmp'

        self.lines = None
        if not os.path.exists(self.edges_filename):
            # Save the edge data in a file, because its too heavy for dynamic memory
            f = open(self.edges_filename, 'w')
            counter = 0
            distances = []
            self.lines = 0
            for vertex in self.image_iterator():
                print("Calculating weights{}   ".format('.' * int(counter / 1000)), end="\r")
                for neighbour in self.neighbours(vertex):
                    dist = self.distance(vertex, neighbour)
                    file_string = "{}:[{},{}][{},{}]\n".format(
                        self.distance(vertex, neighbour),
                        vertex[0], 
                        vertex[1], 
                        neighbour[0], 
                        neighbour[1]
                    )
                    # Write straight away if distance is 0
                    if dist == 0:
                        f.write(file_string)
                    # If distance is not 0, then store in array for later sorting
                    distances.append([dist, file_string])
                    self.lines += 1

                counter = (counter + 1) % 4000
            
            for _, line in sorted(distances, key = lambda elem: elem[0]):
                f.write(line)

            f.close()
            if self.store_data:
                f = open("{}.meta".format(self.edges_filename), 'w')
                f.write("{}".format(self.lines))
                f.close()
        else:
            f = open("{}.meta".format(self.edges_filename), 'r')
            self.lines = int(f.readline())
            f.close()

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
        # Array of weights for this cluster
        weights = []
        # Dictionary of union-find nodes for the algorithm
        mst = {}
        for cluster_vertex in self.clusters[root.coordinates]:
            for neighbour in self.neighbours(cluster_vertex):
                if self.vertices[cluster_vertex].find() == self.vertices[neighbour].find():
                    weights.append([self.distance(cluster_vertex, neighbour), cluster_vertex, neighbour])
            mst[cluster_vertex] = Vertex(cluster_vertex)

        mst_weights = []

        for weight, vertex1, vertex2 in sorted(weights, key=lambda elem: elem[0]):
            # Edge has already been counted
            if mst[vertex1].find() == mst[vertex2].find():
                continue
            # There is no edge connecting the nodes in the mst
            else:
                # Connect the nodes with this edge
                mst[vertex1].find().unite(mst[vertex2].find())
                mst_weights.append(weight)

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
        Deletes the edge information file when the algorithm is over
        '''
        f = open(self.edges_filename, 'r')

        counter = 0

        # Algorithm
        for line in f:
            print("Processing edges {:.2%}".format(counter / self.lines), end="\r")
            weight, vertex1, vertex2 = parse_line(line)
            root1, root2 = self.vertices[vertex1].find(), self.vertices[vertex2].find()
            if (root1 != root2 and
                weight < self.minimum_internal_dif(root1, root2)):
                # Mutate the dictionary to merge the new clusters
                for vertex in self.clusters[root2.coordinates]:
                    self.clusters[root1.coordinates].append(vertex)
                del self.clusters[root2.coordinates]
                # Mutate the union-find to update with the join
                root1.unite(root2)
            counter += 1
        print("Processing edges {:.2%}".format(counter / self.lines), end="\r")

        f.close()

        if not self.store_data:
            # Delete the edge data
            os.remove(self.edges_filename)

        return self.clusters.values()


line_pattern = re.compile(r"(\d*\.\d*):\[(\d*),(\d*)\]\[(\d*),(\d*)\]")
def parse_line(line):
    '''Parses a line of the edge file defined in GraphSegmentator
    -------
    Returns:
        a float, and two tuples
    '''
    match = line_pattern.match(line)
    return float(match.group(1)), (int(match.group(2)), int(match.group(3))), (int(match.group(4)), int(match.group(5)))

if __name__ == '__main__':
    import skimage.io
    img = skimage.io.imread("images/image_1.jpg")
    g = GraphSegmentator(img, 150, store_data=True)
    clusters = g.segment()

    import colorsys
    N = len(clusters)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    new_img = np.zeros(img.shape)

    for index, cluster in enumerate(clusters):
        new_img[cluster] = RGB_tuples[index]

    skimage.io.imsave("result.png", new_img)
    
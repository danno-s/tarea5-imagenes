from numpy.linalg import norm

class Edge():
    def __init__(self, vertex1, vertex2, distance = None):
        '''Stores the two vertices as the two nodes that make up this edge.
        Calculates the distance, but can receive a precalculated distance as well.
        '''
        self.vertex1, self.vertex2 = vertex1, vertex2
        if distance is None:
            self.weight = norm(vertex1 - vertex2)
        else:
            self.weight = distance

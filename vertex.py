class Vertex():
    '''Class that defines a vertex in the graph.
    Used by Cluster to perform union-join operations.
    '''

    def __init__(self, coordinates):
        '''Initializes this vertex with itself as its parent (and root).
        ----------
        Parameters:
            coordinates:
                numpy array of length 2
        '''
        self.coordinates = coordinates
        self.parent = self

    def __eq__(self, other_vertex):
        '''Simple equality based on the coordinates field
        '''
        return self.coordinates == other_vertex.coordinates

    def __ne__(self, other_vertex):
        '''Simple inequality based on the coordinates field
        '''
        return self.coordinates != other_vertex.coordinates

    def __str__(self):
        '''Returns a nicely formatted string representing this vertex
        '''
        if self != self.parent:
            return "Vertex[{}, {}]".format(self.coordinates, self.parent)
        return "Vertex[{}, root]".format(self.coordinates)

    def __hash__(self):
        return self.__str__().__hash__()

    def find(self):
        '''Returns the vertex that acts as the root of this cluster.
        '''
        if self.parent != self:
            self.parent = self.parent.find()
        return self.parent

    def unite(self, new_element):
        '''Changes the given vertex's parent to this
        ----------
        Parameters:
            new_element:
                Vertex to be used as this instance's parent
        '''
        new_element.parent = self
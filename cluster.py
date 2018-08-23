class Cluster():
    '''Defines a set of vertices, and supports union-find operations.
    A vertex in this set is strictly a numpy array of length 2. 
    '''

    def __init__(self, vertex):
        '''Constructor that takes a single vertex as the set.
        ----------
        Parameters:
            vertex:
                the single element contained in this cluster
        '''
        self.vertices = [vertex]

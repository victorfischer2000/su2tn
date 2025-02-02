class MPS(object):
    """
    Matrix product state (MPS) class.

    The i-th MPS tensor has dimension `[d, D[i], D[i+1]]` with `d` the physical
    dimension at each site and `D` the list of virtual bond dimensions.
    """

    def __init__(cls, Alist):
        """
        Create a matrix product state.
        """
        cls.A = Alist
        for i in range(len(cls.A) - 1):
            assert cls.A[i].fusionTree == [[-1, -2, -3]]
            assert cls.A[i].fusionTreeDirections == [-1]

    @property
    def local_dim(cls):
        """Local (physical) dimension at each lattice site."""
        return cls.d

    @property
    def nsites(cls):
        """Number of lattice sites."""
        return len(cls.A)

    def get_full_tensor_state(self):
        """Returns the state as full tensor"""
        pass

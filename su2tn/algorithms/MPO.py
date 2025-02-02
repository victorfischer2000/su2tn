

class MPO(object):
    """
    Matrix product operator (MPO) class.

    The i-th MPO tensor has dimension `[d, d, D[i], D[i+1]]` with `d` the physical
    dimension at each site and `D` the list of virtual bond dimensions.
    """

    def __init__(cls, Alist):
        """
        Create a matrix product operator.
        """
        cls.A = Alist

    @property
    def nsites(cls):
        """Number of lattice sites."""
        return len(cls.A)

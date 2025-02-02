from su2tn.su2_tensor import SU2Tensor
import numpy as np
from su2tn.util import crandn


def return_MPS_list(n_sites, odd_bonds=None, even_bonds=None, fill='random real'):
    assert n_sites % 2 == 0

    if odd_bonds is None:
        odd_bonds = [(1/2, 4), (3/2, 5), (5/2, 3), (7/2, 2)]
    if even_bonds is None:
        even_bonds = [(0, 5), (1, 4), (2, 3), (3, 2), (4, 3)]

    MPSlist = [None for _ in range(n_sites)]
    for i in range(int(n_sites / 2)):
        if i == 0:
            MPSlist[0] = make_MPS(edgeIrreps1=[(1/2, 1)],
                                  edgeIrreps2=[(0, 1)],
                                  edgeIrreps3=odd_bonds[:1], fill=fill)
            MPSlist[n_sites-1] = make_MPS(edgeIrreps1=[(1/2, 1)],
                                          edgeIrreps2=odd_bonds[:1],
                                          edgeIrreps3=[(0, 1)], fill=fill)

        elif i % 2 == 0:
            idx = int(i/2 + 1)

            odds_idx = min(idx, len(odd_bonds))
            odds = odd_bonds[:odds_idx]

            evens_idx = min(idx, len(even_bonds))
            evens = even_bonds[:evens_idx]

            # from left
            MPSlist[i] = make_MPS(edgeIrreps1=[(1 / 2, 1)],
                                  edgeIrreps2=evens,
                                  edgeIrreps3=odds, fill=fill)
            # from right
            MPSlist[int(n_sites - 1 - i)] = make_MPS(edgeIrreps1=[(1 / 2, 1)],
                                                   edgeIrreps2=odds,
                                                   edgeIrreps3=evens, fill=fill)

        elif i % 2 == 1:
            idx = int((i + 1) / 2)

            odds_idx = min(idx, len(odd_bonds))
            odds = odd_bonds[:odds_idx]

            evens_idx = min(idx + 1, len(even_bonds))
            evens = even_bonds[:evens_idx]
            # from left
            MPSlist[i] = make_MPS(edgeIrreps1=[(1 / 2, 1)],
                                  edgeIrreps2=odds,
                                  edgeIrreps3=evens, fill=fill)
            # from right
            MPSlist[int(n_sites - 1 - i)] = make_MPS(edgeIrreps1=[(1 / 2, 1)],
                                                   edgeIrreps2=evens,
                                                   edgeIrreps3=odds, fill=fill)

    verify_MPS_list(MPSlist)
    return MPSlist


def verify_MPS_list(MPS_list):
    for i in range(len(MPS_list)-1):
        left = MPS_list[i]
        right = MPS_list[i+1]

        bondleft = left.listOfOpenEdges[2]
        bondright = right.listOfOpenEdges[1]

        assert bondleft['edgeName'] == -3
        assert bondright['edgeName'] == -2

        assert bondleft['edgeIrreps'] == bondright['edgeIrreps']


def make_MPS(edgeIrreps1, edgeIrreps2, edgeIrreps3, fill):
    """
    Start with a A tensor
    """
    physicalEdge = {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': edgeIrreps1, 'isFused': False,
                    'originalIrreps': None}

    bondEdge1 = {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': edgeIrreps2, 'isFused': False,
                 'originalIrreps': None}

    bondEdge2 = {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': edgeIrreps3, 'isFused': False,
                 'originalIrreps': None}

    A = make_MPS_from_edgeEntries(physicalEdge=physicalEdge, bondEdge1=bondEdge1, bondEdge2=bondEdge2, fill=fill)

    return A


def make_MPS_from_edgeEntries(physicalEdge, bondEdge1, bondEdge2, fill):
    listOfOpenEdges = [physicalEdge, bondEdge1, bondEdge2]

    listOfDegeneracyTensors = []

    fusionTree = [[-1, -2, -3]]
    fusionTreeDirection = [-1]

    A = SU2Tensor(listOfOpenEdges=listOfOpenEdges, listOfDegeneracyTensors=listOfDegeneracyTensors,
                  fusionTree=fusionTree, fusionTreeDirections=fusionTreeDirection)

    for chargeSector in A.listOfChargeSectors:
        j2, j3 = chargeSector[1], chargeSector[2]
        # physical leg dimension
        j1_dim = 1
        # virtual bond dimension
        j2_dim = [edgeIrrep[1] for edgeIrrep in bondEdge1['edgeIrreps'] if edgeIrrep[0] == j2][0]
        j3_dim = [edgeIrrep[1] for edgeIrrep in bondEdge2['edgeIrreps'] if edgeIrrep[0] == j3][0]

        # degTensor = np.array(np.random.randn(j1_dim, j2_dim, j3_dim), dtype='complex128')
        if fill == 'zero':
            degTensor = np.zeros((j1_dim, j2_dim, j3_dim)).astype('complex128')
        elif fill == 'random real':
            # random real entries
            degTensor = np.random.randn(j1_dim, j2_dim, j3_dim).astype('complex128')
        elif fill == 'random complex':
            # random complex entries
            degTensor = crandn((j1_dim, j2_dim, j3_dim))
        else:
            raise ValueError('fill = {} invalid.'.format(fill))
        # degTensor = crandn((j1_dim, j2_dim, j3_dim))
        listOfDegeneracyTensors.append(degTensor)

    A.listOfDegeneracyTensors = listOfDegeneracyTensors

    return A
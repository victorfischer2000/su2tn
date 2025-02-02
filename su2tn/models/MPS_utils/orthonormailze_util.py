import numpy as np
from su2tn.su2_tensor import SU2Tensor

def update_listOfOpenEdges(A, irrepName, irrepValue, new_dim):
    openEdgeEntry = [entry for entry in A.listOfOpenEdges if entry['edgeName'] == irrepName][0]
    A.listOfOpenEdges.remove(openEdgeEntry)

    original_edgeIrreps = openEdgeEntry['edgeIrreps']
    original_edgeIrreps_entry = [edgeIrrep for edgeIrrep in original_edgeIrreps if edgeIrrep[0] == irrepValue][0]
    original_edgeIrreps.remove(original_edgeIrreps_entry)

    original_edgeIrreps_entry = list(original_edgeIrreps_entry)
    original_edgeIrreps_entry[1] = new_dim
    original_edgeIrreps.append(tuple(original_edgeIrreps_entry))

    openEdgeEntry['edgeIrreps'] = original_edgeIrreps
    A.listOfOpenEdges.append(openEdgeEntry)


def get_two_MPS_tensors():
    # first tensor
    fusionTree1 = [[-1, -2, -3]]
    fusionTreeDirections1 = [-1]

    listOpenEdges1 = [
        {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(1 / 2, 2)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(3 / 2, 3)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(1, 4)], 'isFused': False, 'originalIrreps': None}
    ]

    su2tensor1 = SU2Tensor(listOfOpenEdges=listOpenEdges1,
                          listOfDegeneracyTensors=[None for _ in range(len(listOpenEdges1))],
                          fusionTree=fusionTree1,
                          fusionTreeDirections=fusionTreeDirections1)

    listOfDegeneracyTensors1 = []
    for chargeSector in su2tensor1.listOfChargeSectors:
        j1, j2, j3 = chargeSector[0], chargeSector[1], chargeSector[2]
        # physical leg dimension
        assert len([edgeIrrep[1] for edgeIrrep in listOpenEdges1[0]['edgeIrreps'] if edgeIrrep[0] == j1]) == 1
        j1_dim = [edgeIrrep[1] for edgeIrrep in listOpenEdges1[0]['edgeIrreps'] if edgeIrrep[0] == j1][0]
        # virtual bond dimension
        assert len([edgeIrrep[1] for edgeIrrep in listOpenEdges1[1]['edgeIrreps'] if edgeIrrep[0] == j2]) == 1
        j2_dim = [edgeIrrep[1] for edgeIrrep in listOpenEdges1[1]['edgeIrreps'] if edgeIrrep[0] == j2][0]
        assert len([edgeIrrep[1] for edgeIrrep in listOpenEdges1[2]['edgeIrreps'] if edgeIrrep[0] == j3]) == 1
        j3_dim = [edgeIrrep[1] for edgeIrrep in listOpenEdges1[2]['edgeIrreps'] if edgeIrrep[0] == j3][0]
        np.random.seed(1)
        listOfDegeneracyTensors1.append(np.random.rand(j1_dim, j2_dim, j3_dim))
        np.random.seed(None)
    su2tensor1.listOfDegeneracyTensors = listOfDegeneracyTensors1

    # second tensor
    fusionTree2 = [[-1, -2, -3]]
    fusionTreeDirections2 = [-1]

    listOpenEdges2 = [
        {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(1 / 2, 2)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(1, 4)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(3/2, 3)], 'isFused': False, 'originalIrreps': None}
    ]

    su2tensor2 = SU2Tensor(listOfOpenEdges=listOpenEdges2,
                          listOfDegeneracyTensors=[None for _ in range(len(listOpenEdges2))],
                          fusionTree=fusionTree2,
                          fusionTreeDirections=fusionTreeDirections2)

    listOfDegeneracyTensors2 = []
    for chargeSector in su2tensor2.listOfChargeSectors:
        j1, j2, j3 = chargeSector[0], chargeSector[1], chargeSector[2]
        # physical leg dimension
        assert len([edgeIrrep[1] for edgeIrrep in listOpenEdges2[0]['edgeIrreps'] if edgeIrrep[0] == j1]) == 1
        j1_dim = [edgeIrrep[1] for edgeIrrep in listOpenEdges2[0]['edgeIrreps'] if edgeIrrep[0] == j1][0]
        # virtual bond dimension
        assert len([edgeIrrep[1] for edgeIrrep in listOpenEdges2[1]['edgeIrreps'] if edgeIrrep[0] == j2]) == 1
        j2_dim = [edgeIrrep[1] for edgeIrrep in listOpenEdges2[1]['edgeIrreps'] if edgeIrrep[0] == j2][0]
        assert len([edgeIrrep[1] for edgeIrrep in listOpenEdges2[2]['edgeIrreps'] if edgeIrrep[0] == j3]) == 1
        j3_dim = [edgeIrrep[1] for edgeIrrep in listOpenEdges2[2]['edgeIrreps'] if edgeIrrep[0] == j3][0]
        np.random.seed(0)
        listOfDegeneracyTensors2.append(np.random.rand(j1_dim, j2_dim, j3_dim))
        np.random.seed(None)
    su2tensor2.listOfDegeneracyTensors = listOfDegeneracyTensors2

    return su2tensor1, su2tensor2
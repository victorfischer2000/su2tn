from su2tn.su2_tensor import SU2Tensor
import numpy as np


def heisenberg_MPO(n_sites, J):
    """
    Return the SU2 tensors of a MPO for the Heisenberg chain without PBC.
    """
    MPS_tensors = []
    for i in range(n_sites):
        if i == 0:
            MPS_tensors.append(return_Heisenberg_su2_MPO_left(J=J))
        elif i == n_sites - 1:
            MPS_tensors.append(return_Heisenberg_su2_MPO_right(J=J))
        else:
            MPS_tensors.append(return_Heisenberg_su2_MPO_middle(J=J))

    return MPS_tensors


def get_full_MPO_from_su2tensors(MPS_tensors):
    """
    For tests: get the full tensor from the su2-MPS tensors for two sites.
    """
    assert len(MPS_tensors) == 2

    output = SU2Tensor.einsum(su2tensor1=MPS_tensors[0], subscript1=(0, 1, 2, 3),
                              su2tensor2=MPS_tensors[1], subscript2=(4, 2, 5, 6))

    assert len(list(output.return_explicit_tensor_blocks().values())) == 1
    tensor = list(output.return_explicit_tensor_blocks().values())[0]

    tensor = np.trace(tensor, axis1=1, axis2=4)

    tensor = np.transpose(tensor, (0, 2, 1, 3))
    return tensor


def return_Heisenberg_su2_MPO_middle(J):
    fusionTree = [[-1, -2, 1], [1, -3, -4]]
    fusionTreeDirections = [-1, +1]

    listOfOpenEdges = [
        {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(1/2, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(0, 2), (1, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(0, 2), (1, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -4, 'edgeNumber': 4, 'edgeIrreps': [(1/2, 1)], 'isFused': False, 'originalIrreps': None}
    ]

    listOfChargeSectors = [[1/2, 1/2, 0, 0, 1/2], [1/2, 1/2, 0, 1, 1/2], [1/2, 1/2, 1, 0, 1/2]]

    listOfDegeneracyTensors = [np.reshape(np.array([[1, 0], [0, 1]]), (1, 2, 2, 1)),
                               np.reshape(np.array([-1, 0]), (1, 2, 1, 1)),
                               np.reshape(np.array([0, J * (0.5 * (0.5 + 1))]), (1, 1, 2, 1))]

    return SU2Tensor(listOfOpenEdges=listOfOpenEdges, listOfDegeneracyTensors=listOfDegeneracyTensors,
                     fusionTree=fusionTree, fusionTreeDirections=fusionTreeDirections,
                     listOfChargeSectors=listOfChargeSectors)


def return_Heisenberg_su2_MPO_left(J):
    fusionTree = [[-1, -2, 1], [1, -3, -4]]
    fusionTreeDirections = [-1, +1]

    listOfOpenEdges = [
        {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(1/2, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(0, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(0, 2), (1, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -4, 'edgeNumber': 4, 'edgeIrreps': [(1/2, 1)], 'isFused': False, 'originalIrreps': None}
    ]

    listOfChargeSectors = [[1/2, 1/2, 0, 0, 1/2], [1/2, 1/2, 0, 1, 1/2]]

    listOfDegeneracyTensors = [np.reshape(np.array([1, 0]), (1, 1, 2, 1)),
                               np.reshape(np.array([-1]), (1, 1, 1, 1))]

    return SU2Tensor(listOfOpenEdges=listOfOpenEdges, listOfDegeneracyTensors=listOfDegeneracyTensors,
                     fusionTree=fusionTree, fusionTreeDirections=fusionTreeDirections,
                     listOfChargeSectors=listOfChargeSectors)


def return_Heisenberg_su2_MPO_right(J):
    fusionTree = [[-1, -2, 1], [1, -3, -4]]
    fusionTreeDirections = [-1, +1]

    listOfOpenEdges = [
        {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(1/2, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(0, 2), (1, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(0, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -4, 'edgeNumber': 4, 'edgeIrreps': [(1/2, 1)], 'isFused': False, 'originalIrreps': None}
    ]

    listOfChargeSectors = [[1/2, 1/2, 0, 0, 1/2], [1/2, 1/2, 1, 0, 1/2]]

    listOfDegeneracyTensors = [np.reshape(np.array([[0, 1]]), (1, 2, 1, 1)),
                               np.reshape(np.array([J * (0.5 * (0.5 + 1))]), (1, 1, 1, 1))]

    return SU2Tensor(listOfOpenEdges=listOfOpenEdges, listOfDegeneracyTensors=listOfDegeneracyTensors,
                     fusionTree=fusionTree, fusionTreeDirections=fusionTreeDirections,
                     listOfChargeSectors=listOfChargeSectors)


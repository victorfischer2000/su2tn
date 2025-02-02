import numpy as np
import pandas as pd
import copy

from su2tn.su2_tensor import SU2Tensor


def conjugate_MPS(A: SU2Tensor):
    """
    Return the conjugated tensor for the MPS A.
    """
    A_copy = copy.copy(A)
    A_copy.reverse_left_most_leg(reverseIrrepName=-1)
    A_copy.reverse_right_most_leg(reverseIrrepName=-3)
    A_copy.perform_permutation(irrep1Name=-2, irrep2Name=-3)
    A_copy.reverse_right_most_leg(reverseIrrepName=-2)

    for idx, degTensor in enumerate(A_copy.listOfDegeneracyTensors):
        A_copy.listOfDegeneracyTensors[idx] = np.conj(degTensor)

    order_in_nameOrdering_conjugation(A_copy)

    return A_copy


def order_in_nameOrdering_conjugation(A_target):
    """
    After reversing back, the nameOrdering is [-1, -3, -2]. We change it back to [-1, -2, -3].
    """
    assert A_target.nameOrdering == [-1, -3, -2]

    df_chargeSectors = pd.DataFrame(A_target.listOfChargeSectors, columns=A_target.nameOrdering)
    # switch the columns of the irreps 1 and 2
    newChargeSectors = df_chargeSectors[[-3, -1, -2]].values.tolist()
    A_target.listOfChargeSectors = newChargeSectors

    A_target.nameOrdering = [-1, -2, -3]
    openEdge1 = [openEdge for openEdge in A_target.listOfOpenEdges if openEdge['edgeName'] == -1][0].copy()
    openEdge2 = [openEdge for openEdge in A_target.listOfOpenEdges if openEdge['edgeName'] == -2][0].copy()
    openEdge3 = [openEdge for openEdge in A_target.listOfOpenEdges if openEdge['edgeName'] == -3][0].copy()

    openEdge1['edgeName'], openEdge1['edgeNumber'] = -2, 2
    openEdge2['edgeName'], openEdge2['edgeNumber'] = -3, 3
    openEdge3['edgeName'], openEdge3['edgeNumber'] = -1, 1
    newListOfOpenEdges = [openEdge3, openEdge1, openEdge2]
    A_target.listOfOpenEdges = newListOfOpenEdges

    A_target.fusionTree = [[-1, -2, -3]]
    A_target.fusionTreeDirections = [+1]

    for idx, entry in enumerate(A_target.listOfDegeneracyTensors):
        A_target.listOfDegeneracyTensors[idx] = np.transpose(np.conj(A_target.listOfDegeneracyTensors[idx]), (1, 0, 2))


def conjugate_singe_fusionNode_tensor(A):
    """
    Conjugate a tensor that has a single fusion node a underlying fusion tree, e.g. an MPS tensor.
    """
    assert A.fusionTreeDirections == [-1]

    newFusionTree = [[-1, -2, -3]]
    newFusionTreeDirections = [+1]

    openEdge1 = [openEdge for openEdge in A.listOfOpenEdges if openEdge['edgeName'] == -1][0].copy()
    openEdge2 = [openEdge for openEdge in A.listOfOpenEdges if openEdge['edgeName'] == -2][0].copy()
    openEdge3 = [openEdge for openEdge in A.listOfOpenEdges if openEdge['edgeName'] == -3][0].copy()

    openEdge1['edgeName'], openEdge1['edgeNumber'] = -2, 2
    openEdge2['edgeName'], openEdge2['edgeNumber'] = -3, 3
    openEdge3['edgeName'], openEdge3['edgeNumber'] = -1, 1
    newListOfOpenEdges = [openEdge3, openEdge1, openEdge2]

    newListOfChargeSectors = pd.DataFrame(A.listOfChargeSectors.copy(), columns=A.nameOrdering)
    newListOfChargeSectors = newListOfChargeSectors[[-3, -1, -2]].values.tolist()

    newNameOrdering = [-1, -2, -3]

    newListOfDegeneracyTensors = []
    for degTensor in A.listOfDegeneracyTensors:
        newListOfDegeneracyTensors.append(np.transpose(np.conj(degTensor), (2, 0, 1)))

    return SU2Tensor(fusionTree=newFusionTree, fusionTreeDirections=newFusionTreeDirections,
                     listOfDegeneracyTensors=newListOfDegeneracyTensors, listOfChargeSectors=newListOfChargeSectors,
                     nameOrdering=newNameOrdering, listOfOpenEdges=newListOfOpenEdges)


def norm_of_single_fusionNode_tensor(su2tensor):
    """
    Return the norm a tensor that has a single fusion node as underlying fusion tree, e.g., an MPS tensor.
    """
    assert su2tensor.fusionTreeDirections == [-1]

    vdot_out = 0

    for chargeSector in su2tensor.listOfChargeSectors:
        jab = chargeSector[2]
        AdegTensor = su2tensor.listOfDegeneracyTensors[su2tensor.listOfChargeSectors.index(chargeSector)]

        vdot_out += np.tensordot(np.conj(AdegTensor), AdegTensor, ((0, 1, 2), (0, 1, 2))) * (2 * jab + 1)

    return np.sqrt(vdot_out)


def vdot_of_two_single_fusionNodeode_tensors(A, B):
    """
    Calculate the vector product of two tensors that each have a single fusion node as underlying fusion tree, e.g.,
    two MPS tensors.
    """
    vdot_out = 0
    for chargeSector in A.listOfChargeSectors:
        jab = chargeSector[2]
        AdegTensor = A.listOfDegeneracyTensors[A.listOfChargeSectors.index(chargeSector)]

        BdegTensor = B.listOfDegeneracyTensors[B.listOfChargeSectors.index(chargeSector)]

        vdot_out += np.tensordot(np.conj(AdegTensor), BdegTensor, ((0, 1, 2), (0, 1, 2))) * (2 * jab + 1)

    return vdot_out



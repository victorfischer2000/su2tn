import numpy as np
import pandas as pd
from su2tn.su2tensor_utils.switch_fusion_tree_make_neighbors import switch_fusion_tree_to_make_neighboring


def perform_permute(su2tensor, irrep1, irrep2):
    """
    Performs a swapping operation on irrep1 and irrep2 of the su2tensor.
    """
    assert irrep1 < 0 and irrep2 < 0
    # switch the fusion tree to prepare for switching the legs.
    switch_fusion_tree_to_make_neighboring(su2tensor=su2tensor, irrep1=irrep1, irrep2=irrep2)

    # get the added irrep
    irrep12 = permute_return_added_irrep(fusionTree=su2tensor.fusionTree,
                                         fusionTreeDirections=su2tensor.fusionTreeDirections,
                                         irrep1=irrep1, irrep2=irrep2)

    newFusionTree = permute_change_fusion_tree(oldFusionTree=su2tensor.fusionTree, irrep1=irrep1, irrep2=irrep2)

    nameOrdering = [0] + su2tensor.nameOrdering
    oldListOfChargeSectors = su2tensor.listOfChargeSectors.copy()
    for idx in range(len(oldListOfChargeSectors)):
        oldListOfChargeSectors[idx] = [0] + oldListOfChargeSectors[idx]

    # update the charge sectors
    newNameOrdering, newChargeSectors = permute_change_charge_sectors(oldChargeSectors=oldListOfChargeSectors,
                                                                      nameOrdering=nameOrdering,
                                                                      irrep1=irrep1, irrep2=irrep2)

    # update the degeneracy tensors
    newListOfDegeneracyTensors = permute_get_new_degeneracy_tensors(chargeSectors=newChargeSectors,
                                                                    nameOrdering=newNameOrdering,
                                                                    listOfDegeneracyTensors=su2tensor.listOfDegeneracyTensors,
                                                                    irrep1=irrep1,
                                                                    irrep2=irrep2,
                                                                    irrep12=irrep12)
    su2tensor.listOfDegeneracyTensors = newListOfDegeneracyTensors

    newListOfOpenEdges = permute_update_listOfOpenEdges(su2tensor=su2tensor,
                                                        irrep1Name=irrep1, irrep2Name=irrep2)
    su2tensor.listOfOpenEdges = newListOfOpenEdges

    newNameOrdering = newNameOrdering[1:]
    newChargeSectors = np.array(newChargeSectors)[:, 1:].tolist()

    su2tensor.nameOrdering = newNameOrdering
    su2tensor.listOfChargeSectors = newChargeSectors

    su2tensor.fusionTree = newFusionTree


def R_factor(ja, jb, jab):
    """
    Returns the R-factor resulting from swapping two indices.
    """
    return (-1)**(ja + jb - jab)


def permute_change_fusion_tree(oldFusionTree, irrep1, irrep2):
    """
    Switches the labels of the irrep1 and irrep2 in the fusionTree nodes list. The order of nodes stays the same.
    """
    newFusionTree = pd.DataFrame(oldFusionTree.copy())
    # switch irrep 1 and 2
    newFusionTree = newFusionTree.replace([irrep1, irrep2], [irrep2, irrep1])

    return newFusionTree.values.tolist()


def permute_change_charge_sectors(oldChargeSectors, nameOrdering, irrep1, irrep2):
    """
    Switches the order in the chargeSectors to match the new charge sectors after the switching.
    """
    df_oldChargeSectors = pd.DataFrame(oldChargeSectors.copy(), columns=nameOrdering)
    newNameOrdering = pd.DataFrame(nameOrdering.copy())
    newNameOrdering = newNameOrdering.replace([irrep1, irrep2], [irrep2, irrep1]).values.flatten().tolist()
    # switch the columns of the irreps 1 and 2
    newChargeSectors = df_oldChargeSectors[newNameOrdering]

    return newNameOrdering, newChargeSectors.values.tolist()


def permute_return_added_irrep(fusionTree, fusionTreeDirections, irrep1, irrep2):
    """
    When swapping two irreps, they are part of the same node and are both either
    """
    permuteNode = [node for node in fusionTree if (irrep1 in node) and (irrep2 in node)]
    assert len(permuteNode) == 1
    permuteNode = permuteNode[0]
    permuteNode_idx = fusionTree.index(permuteNode)
    permuteNodeDirection = fusionTreeDirections[permuteNode_idx]

    # check that both legs are either incoming or outgoing
    if permuteNodeDirection == -1:
        assert ((permuteNode[0] == irrep1 and permuteNode[1] == irrep2)
                or (permuteNode[0] == irrep2 and permuteNode[1] == irrep1))
    elif permuteNodeDirection == 1:
        assert ((permuteNode[1] == irrep1 and permuteNode[2] == irrep2)
                or (permuteNode[1] == irrep2 and permuteNode[2] == irrep1))

    # get the name of the added irrep
    if permuteNodeDirection == -1:
        irrep12 = permuteNode[2]
    elif permuteNodeDirection == 1:
        irrep12 = permuteNode[0]

    return irrep12


def permute_get_new_degeneracy_tensors(chargeSectors, nameOrdering, listOfDegeneracyTensors, irrep1, irrep2, irrep12):
    """
    Get the new degeneracy tensors
    """
    newListOfDegeneracyTensors = []
    irrep1_idx = nameOrdering.index(irrep1)
    irrep2_idx = nameOrdering.index(irrep2)
    irrep12_idx = nameOrdering.index(irrep12)

    outerLegOrder = list(np.array(nameOrdering)[np.array(nameOrdering) < 0])
    degTensor_idx1 = outerLegOrder.index(irrep1)
    degTensor_idx2 = outerLegOrder.index(irrep2)

    axes = list(range(len(outerLegOrder)))
    axes[degTensor_idx1] = degTensor_idx2
    axes[degTensor_idx2] = degTensor_idx1

    for chargeSector, degeneracyTensor in zip(chargeSectors, listOfDegeneracyTensors):
        j1 = chargeSector[irrep1_idx]
        j2 = chargeSector[irrep2_idx]
        j12 = chargeSector[irrep12_idx]

        # do the permutation
        degeneracyTensor = np.transpose(degeneracyTensor, axes=axes)

        # multiply the R factor
        # newListOfDegeneracyTensors.append(R_factor(j1, j2, j12) * degeneracyTensor)
        newListOfDegeneracyTensors.append(degeneracyTensor)

    return newListOfDegeneracyTensors


def permute_update_listOfOpenEdges(su2tensor, irrep1Name, irrep2Name):
    """
    Updates the listOfOpenEdges so that the edgeNumber is switched between the two permuted legs.
    """
    listOfOpenEdges = su2tensor.listOfOpenEdges.copy()
    openLeg1 = [ol for ol in listOfOpenEdges if ol['edgeName'] == irrep1Name][0]
    listOfOpenEdges.remove(openLeg1)

    openLeg2 = [ol for ol in listOfOpenEdges if ol['edgeName'] == irrep2Name][0]
    listOfOpenEdges.remove(openLeg2)

    edgeNumber1 = openLeg1['edgeNumber']
    edgeNumber2 = openLeg2['edgeNumber']
    openLeg1['edgeNumber'] = edgeNumber2
    openLeg2['edgeNumber'] = edgeNumber1

    listOfOpenEdges.append(openLeg1)
    listOfOpenEdges.append(openLeg2)

    return listOfOpenEdges


def permute_bubble_sort(axes):
    """
    Finds a sequence to pairwise permute the axis to the desired permutation.
    """
    n = len(axes)
    axes = list(axes)
    steps = []
    for j in range(n):
        for i in range(n - j - 1):
            if axes[i] > axes[i + 1]:
                axes[i], axes[i+1] = axes[i+1], axes[i]
                steps.append([axes[i], axes[i+1]])

    return steps


def transpose(su2tensor, axes):
    """
    Transposes the su2tensor. Analogous to numpy method transpose. Uses bubble sort algorithm to find a good way of
    applying the permutations.
    """
    assert len(axes) == su2tensor.numberOfOpenEdges
    saveNameOpenEdges = su2tensor.nameOrdering[su2tensor.numberOfInternalEdges:]
    steps = permute_bubble_sort(axes)

    for step in reversed(steps):
        irrep1 = saveNameOpenEdges[step[0]]
        irrep2 = saveNameOpenEdges[step[1]]
        perform_permute(su2tensor, irrep1, irrep2)


if __name__ == '__main__':
    from su2tn.su2_tensor import SU2Tensor

    fusionTree = [[-1, -2, -3]]
    fusionTreeDirections = [-1]

    listOpenEdges = [
        {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None}
    ]

    su2tensor = SU2Tensor(listOfOpenEdges=listOpenEdges,
                          listOfDegeneracyTensors=[None for _ in range(len(listOpenEdges))],
                          fusionTree=fusionTree,
                          fusionTreeDirections=fusionTreeDirections)

    listOfDegeneracyTensors = []
    for chargeSector in su2tensor.listOfChargeSectors:
        listOfDegeneracyTensors.append(np.random.rand(2, 2, 2))
    su2tensor.listOfDegeneracyTensors = listOfDegeneracyTensors

    transpose(su2tensor=su2tensor, axes=(1, 0, 2))


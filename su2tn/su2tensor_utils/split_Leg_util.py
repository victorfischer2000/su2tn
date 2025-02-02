import numpy as np
import pandas as pd
from copy import deepcopy
from su2tn.su2tensor_utils.calc_charge_sectors_util import calculate_all_charge_sectors
from su2tn.su2tensor_utils.remove_unnec_dummylegs_split import remove_unnecessary_nodes_split


def split_leg(su2tensor, splitLegName):
    # do the checks to make sure that the given leg can be split

    splitLegEntry = split_leg_checks(su2tensor=su2tensor, splitLegName=splitLegName)

    newFusionTree, newFusionTreeDirections = split_legs_build_fusionTree(su2tensor=su2tensor,
                                                                        splitLegName=splitLegName,
                                                                        splitLegEntry=splitLegEntry)

    newListOfOpenEdges, newNameOrdering = split_leg_make_listOfOpenEdges(su2tensor=su2tensor,
                                                                         splitLegEntry=splitLegEntry)

    newChargeSectors, newListOfDegeneracyTensors = split_leg_cut_degeneracyTensors(su2tensor=su2tensor,
                                                                                   splitLegEntry=splitLegEntry,
                                                                                   newListOfOpenEdges=newListOfOpenEdges,
                                                                                   newNameOrdering=newNameOrdering,
                                                                                   newFusionTree=newFusionTree,
                                                                                   newFusionTreeDirections=newFusionTreeDirections)

    newFusionTree, newFusionTreeDirections, newNameOrdering, newChargeSectors = remove_unnecessary_nodes_split(
        chargeSectors=newChargeSectors,
        nameOrdering=newNameOrdering,
        fusionTree=newFusionTree,
        fusionTreeDirections=newFusionTreeDirections
    )

    # do all the changes to the su2tensor
    split_legs_updatesu2tensor_inplace(su2tensor=su2tensor,
                                       newFusionTree=newFusionTree,
                                       newFusionTreeDirections=newFusionTreeDirections,
                                       newNameOrdering=newNameOrdering,
                                       newListOfOpenEdges=newListOfOpenEdges,
                                       newChargeSectors=newChargeSectors,
                                       newListOfDegeneracyTensors=newListOfDegeneracyTensors)


def split_legs_updatesu2tensor_inplace(su2tensor,
                                       newFusionTree, newFusionTreeDirections,
                                       newListOfOpenEdges,
                                       newNameOrdering,
                                       newChargeSectors, newListOfDegeneracyTensors):
    """
    Helper function that updates all the relevant information in the su2tensor inplace.
    """
    su2tensor.fusionTree, su2tensor.fusionTreeDirections = newFusionTree, newFusionTreeDirections
    su2tensor.listOfOpenEdges = newListOfOpenEdges
    su2tensor.nameOrdering = newNameOrdering
    su2tensor.listOfChargeSectors = newChargeSectors
    su2tensor.listOfDegeneracyTensors = newListOfDegeneracyTensors
    su2tensor.listOfStructuralTensors = None
    su2tensor.numberOfOpenEdges = len(su2tensor.listOfOpenEdges)
    su2tensor.numberOfInternalEdges = len(su2tensor.fusionTree) - 1



def split_leg_checks(su2tensor, splitLegName):
    listOfOpenEdges = su2tensor.listOfOpenEdges
    splitLeg = [openEdge for openEdge in listOfOpenEdges if openEdge['edgeName'] == splitLegName]
    assert len(splitLeg) == 1
    assert splitLeg[0]['isFused']

    return splitLeg[0]


def split_legs_build_fusionTree(su2tensor, splitLegName, splitLegEntry):
    """
    Helper function that builds the new fusion tree with splitLegName leg split in two legs.
    """
    newFusionTree = su2tensor.fusionTree.copy()
    newFusionTreeDirections = su2tensor.fusionTreeDirections.copy()

    splitNode = [node for node in newFusionTree if splitLegName in node]
    assert len(splitNode) == 1
    splitNode = deepcopy(splitNode[0])
    splitNode_idx = newFusionTree.index(splitNode)
    splitNodeDirection = su2tensor.fusionTreeDirections[splitNode_idx]

    irrep1Name = splitLegEntry['originalIrreps'][1][0]
    irrep2Name = splitLegEntry['originalIrreps'][2][0]

    newFusionTree[splitNode_idx][splitNode.index(splitLegName)] = su2tensor.numberOfInternalEdges + 1
    # node is a fusion node
    if splitNodeDirection == -1:
        if splitNode[2] == splitLegName:
            newFusionTree.append([su2tensor.numberOfInternalEdges + 1, irrep2Name, irrep1Name])
            newFusionTreeDirections.append(+1)
        else:
            newFusionTree.append([irrep1Name, irrep2Name, su2tensor.numberOfInternalEdges + 1])
            newFusionTreeDirections.append(-1)

    else:
        if splitNode[0] == splitLegName:
            newFusionTree.append([irrep1Name, irrep2Name, su2tensor.numberOfInternalEdges + 1])
            newFusionTreeDirections.append(-1)
        else:
            newFusionTree.append([su2tensor.numberOfInternalEdges + 1, irrep2Name, irrep1Name])
            newFusionTreeDirections.append(+1)

    return newFusionTree, newFusionTreeDirections


def split_leg_make_listOfOpenEdges(su2tensor, splitLegEntry):
    """
    Helper function for splitting a leg that generates the new listOfOpenEdges and nameOrdering.
    """
    newListOfOpenEdges = su2tensor.listOfOpenEdges.copy()
    newListOfOpenEdges.remove(splitLegEntry)

    openLeg1, openLeg2 = {}, {}
    openLeg1['edgeName'] = splitLegEntry['originalIrreps'][1][0]
    openLeg2['edgeName'] = splitLegEntry['originalIrreps'][2][0]

    openLeg1['edgeNumber'] = splitLegEntry['edgeNumber']
    openLeg2['edgeNumber'] = splitLegEntry['edgeNumber'] + 1

    if type(splitLegEntry['originalIrreps'][1][1][0][1]) == list:
        isFused1 = True
        edgeIrreps1 = splitLegEntry['originalIrreps'][1][1][0][1]
        originalIrreps1 = splitLegEntry['originalIrreps'][1][1]
    # type is number
    else:
        isFused1 = False
        edgeIrreps1 = splitLegEntry['originalIrreps'][1][1]
        originalIrreps1 = None

    if type(splitLegEntry['originalIrreps'][2][1][0][1]) == list:
        isFused2 = True
        edgeIrreps2 = splitLegEntry['originalIrreps'][2][1][0][1]
        originalIrreps2 = splitLegEntry['originalIrreps'][2][1]
    # type is number
    else:
        isFused2 = False
        edgeIrreps2 = splitLegEntry['originalIrreps'][2][1]
        originalIrreps2 = None

    openLeg1['edgeIrreps'] = edgeIrreps1
    openLeg2['edgeIrreps'] = edgeIrreps2

    openLeg1['isFused'] = isFused1
    openLeg2['isFused'] = isFused2

    openLeg1['originalIrreps'] = originalIrreps1
    openLeg2['originalIrreps'] = originalIrreps2

    newOpenLegNameOrdering = [None for _ in range(su2tensor.numberOfOpenEdges+1)]

    for ol in newListOfOpenEdges:
        if ol['edgeNumber'] >= max(openLeg1['edgeNumber'], openLeg2['edgeNumber']):
            ol['edgeNumber'] = ol['edgeNumber'] + 1
        newOpenLegNameOrdering[ol['edgeNumber'] - 1] = ol['edgeName']

    newOpenLegNameOrdering[max(openLeg1['edgeNumber'], openLeg2['edgeNumber']) - 1] = openLeg2['edgeName']
    newOpenLegNameOrdering[min(openLeg1['edgeNumber'], openLeg2['edgeNumber']) - 1] = openLeg1['edgeName']
    newNameOrdering = [intLeg for intLeg in range(1, su2tensor.numberOfInternalEdges+2)] + newOpenLegNameOrdering

    newListOfOpenEdges.append(openLeg1)
    newListOfOpenEdges.append(openLeg2)

    return newListOfOpenEdges, newNameOrdering


def split_leg_cut_degeneracyTensors(su2tensor, splitLegEntry,
                                    newListOfOpenEdges, newNameOrdering,
                                    newFusionTree, newFusionTreeDirections):
    """
    Helper function
    """
    splitLegName = splitLegEntry['edgeName']

    oldChargeSectors = su2tensor.listOfChargeSectors.copy()
    newChargeSectors = calculate_all_charge_sectors(listOfOpenEdges=newListOfOpenEdges, nameOrdering=newNameOrdering,
                                                    fusionTree=newFusionTree,
                                                    fusionTreeDirections=newFusionTreeDirections)

    oldListOfDegenercyTensors = su2tensor.listOfDegeneracyTensors.copy()

    groupingNamesList = su2tensor.nameOrdering.copy()

    groupingNamesList[su2tensor.nameOrdering.index(splitLegName)] = su2tensor.numberOfInternalEdges + 1

    newCs_df = pd.DataFrame(newChargeSectors, columns=newNameOrdering)

    newCs_groups = newCs_df.groupby(groupingNamesList)

    newListOfDegeneracyTensors = [None for _ in range(len(newChargeSectors))]

    irrep1Name = splitLegEntry['originalIrreps'][1][0]
    irrep2Name = splitLegEntry['originalIrreps'][2][0]
    edgeIrreps1 = [entry['edgeIrreps'] for entry in newListOfOpenEdges if entry['edgeName'] == irrep1Name][0]
    edgeIrreps2 = [entry['edgeIrreps'] for entry in newListOfOpenEdges if entry['edgeName'] == irrep2Name][0]

    openEdgeOrder = su2tensor.nameOrdering[su2tensor.numberOfInternalEdges:]
    irrep12_idx = openEdgeOrder.index(splitLegName)
    for idx, chargeSector in enumerate(oldChargeSectors):
        oldDegeneracyTensor = oldListOfDegenercyTensors[idx]

        newContributingChargeSectors = newCs_groups.get_group(tuple(chargeSector))[[irrep1Name, irrep2Name]]
        newContributingChargeSectors = newContributingChargeSectors.sort_values(by=[irrep1Name, irrep2Name],
                                                                                ascending=True)

        splitInfo = []
        for idx, row in newContributingChargeSectors.iterrows():

            degDimInfo1 = [irrep for irrep in edgeIrreps1 if irrep[0] == row.values[0]][0]
            degDimInfo2 = [irrep for irrep in edgeIrreps2 if irrep[0] == row.values[1]][0]

            splitInfo.append([idx, degDimInfo1, degDimInfo2])

        splitSections = []
        splitIdx = 0
        for splitInfoEntry in splitInfo:
            splitIdx += splitInfoEntry[1][1] * splitInfoEntry[2][1]
            splitSections.append(splitIdx)

        newDegeneracyTensors = np.split(oldDegeneracyTensor, indices_or_sections=splitSections, axis=irrep12_idx)

        for splitInfoEntry, newDegeneracyTensor in zip(splitInfo, newDegeneracyTensors):
            newShape = list(newDegeneracyTensor.shape)
            newShape[irrep12_idx] = splitInfoEntry[1][1]
            # TODO: Check if this inserts at correct index
            newShape.insert(irrep12_idx+1, splitInfoEntry[2][1])

            newListOfDegeneracyTensors[splitInfoEntry[0]] = np.reshape(newDegeneracyTensor, newShape)

    return newChargeSectors, newListOfDegeneracyTensors


if __name__ == "__main__":
    from su2tn.su2_tensor import SU2Tensor
    from su2tn.su2tensor_utils.fuse_neighborLegs import fuse_neighboring_legs
    fusionTree = [[-1, -2, 1], [1, -3, 2], [2, -4, -5]]
    fusionTreeDirections = [-1, -1, -1]

    listOpenEdges = [
        {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -4, 'edgeNumber': 4, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -5, 'edgeNumber': 5, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None}
    ]

    # fusionTree = [[-1, -2, 1], [1, -3, -4]]
    # fusionTreeDirections = [-1, -1]
    #
    # listOpenEdges = [
    #     {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
    #     {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
    #     {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
    #     {'edgeName': -4, 'edgeNumber': 4, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None}
    # ]

    su2tensor = SU2Tensor(listOfOpenEdges=listOpenEdges,
                          listOfDegeneracyTensors=[None for _ in range(len(listOpenEdges))],
                          fusionTree=fusionTree,
                          fusionTreeDirections=fusionTreeDirections)

    listOfDegeneracyTensors = []
    for chargeSector in su2tensor.listOfChargeSectors:
        listOfDegeneracyTensors.append(np.random.rand(2, 2, 2, 2))
    su2tensor.listOfDegeneracyTensors = listOfDegeneracyTensors

    # calculate the explicit tensors
    control_dict = su2tensor.return_explicit_tensor_blocks()

    # print(su2tensor.listOfChargeSectors)

    fuse_neighboring_legs(su2tensor, -1, -2)
    # print(su2tensor.listOfChargeSectors)
    fuse_neighboring_legs(su2tensor, -1, -3)
    # print(su2tensor.fusionTree)
    splitLeg = [openLeg for openLeg in su2tensor.listOfOpenEdges if openLeg['edgeName'] == -1][0]

    # print(splitLeg)
    # print(split_leg_make_listOfOpenEdges(su2tensor, splitLeg)[1])
    newListOfDegTensors = split_leg(su2tensor, -1)
    print(su2tensor.fusionTree)
    newListOfDegTensors = split_leg(su2tensor, -1)
    print(sorted(list(np.array(newListOfDegTensors).tolist())) == sorted(list(np.array(listOfDegeneracyTensors).tolist())))
    # assert sorted(newListOfDegTensors) == sorted(listOfDegeneracyTensors)


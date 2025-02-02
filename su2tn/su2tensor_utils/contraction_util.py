import numpy as np
import pandas as pd
from collections import Counter
from su2tn.su2tensor_utils.contract_remove_unnec_nodes import remove_unnecessary_nodes_twoDummies


def einsum(su2tensor1, subscript1, su2tensor2, subscript2):
    """
    Function implementing the contracts, similar to Numpy's einsum function.
    """
    assert len(subscript1) == len(su2tensor1.listOfOpenEdges)
    assert len(subscript2) == len(su2tensor2.listOfOpenEdges)

    einsumOrdering, element_counts = einsum_get_einsumOrdering(subscript1=subscript1,
                                                               subscript2=subscript2)

    matchIrrepsOnLegs, contractLegList1, contractLegList2 = einsum_get_matchIrrepsOnLegs(su2tensor1=su2tensor1,
                                                                                         subscript1=subscript1,
                                                                                         su2tensor2=su2tensor2,
                                                                                         subscript2=subscript2,
                                                                                         element_counts=element_counts)

    openLegReplacements, newListOfOpenEdges = einsum_get_openLegReplacements(su2tensor1=su2tensor1,
                                                                             su2tensor2=su2tensor2,
                                                                             contractLegList1=contractLegList1,
                                                                             contractLegList2=contractLegList2)

    (newFusionTree, newFusionTreeDirections, matchIrrepsOnLegs, matchingLegsIsOpen,
     internalLegReplacements, openLegReplacements, dropIrreps) = (
        einsum_getNewFusionTree(matchIrrepsOnLegs=matchIrrepsOnLegs,
                                openLegReplacements=openLegReplacements,
                                oldFusionTree1=su2tensor1.fusionTree.copy(),
                                oldFusionTreeDirection1=su2tensor1.fusionTreeDirections.copy(),
                                oldFusionTree2=su2tensor2.fusionTree.copy(),
                                oldFusionTreeDirection2=su2tensor2.fusionTreeDirections.copy(),
                                numberOfInternalEdges1=su2tensor1.numberOfInternalEdges,
                                numberOfInternalEdges2=su2tensor2.numberOfInternalEdges))

    newNameOrdering1, newNameOrdering2 = (
        einsum_newNameOrderingsForFusionTrees(oldNameOrdering1=su2tensor1.nameOrdering,
                                              oldNameOrdering2=su2tensor2.nameOrdering,
                                              openLegReplacements=openLegReplacements,
                                              internalLegReplacements=internalLegReplacements))

    newNameOrdering, newChargeSectors, newDegeracyTensors = (
        einsum_getNewChargeSectors(oldChargeSectors1=su2tensor1.listOfChargeSectors.copy(),
                                   oldChargeSectors2=su2tensor2.listOfChargeSectors.copy(),
                                   newNameOrdering1=newNameOrdering1,
                                   newNameOrdering2=newNameOrdering2,
                                   matchIrrepsOnLegs=matchIrrepsOnLegs,
                                   oldListOfDegeneracyTensors1=su2tensor1.listOfDegeneracyTensors,
                                   oldListOfDegeneracyTensors2=su2tensor2.listOfDegeneracyTensors,
                                   einsumOrdering=einsumOrdering,
                                   dropIrreps=dropIrreps))

    newFusionTree, newFusionTreeDirections, newNameOrdering, newChargeSectors = remove_unnecessary_nodes_twoDummies(
        chargeSectors=newChargeSectors,
        nameOrdering=newNameOrdering,
        fusionTree=newFusionTree,
        fusionTreeDirections=newFusionTreeDirections
    )

    newFusionTree, newNameOrdering = einsum_makeinternalEdgesLabelingCorrect(newFusionTree=newFusionTree,
                                                                             newNameOrdering=newNameOrdering)

    return (newListOfOpenEdges, newNameOrdering, newChargeSectors, newDegeracyTensors,
            newFusionTree, newFusionTreeDirections)


def einsum_get_einsumOrdering(subscript1, subscript2):
    combined_subscripts = subscript1 + subscript2
    element_counts = Counter(combined_subscripts)

    subscriptOut = tuple([element for element in combined_subscripts if element_counts[element] == 1])
    einsumOrdering = (subscript1, subscript2, subscriptOut)

    return einsumOrdering, element_counts


def einsum_get_matchIrrepsOnLegs(su2tensor1, subscript1, su2tensor2, subscript2, element_counts):
    """
    Check which edge irreps need to match for the contracted tensors.
    """
    openEdgeOrdering1 = [edge for edge in su2tensor1.nameOrdering if edge < 0]
    openEdgeOrdering2 = [edge for edge in su2tensor2.nameOrdering if edge < 0]

    matchIrrepsOnLegs = []
    contractLegList1 = []
    contractLegList2 = []

    for subscript_part in element_counts:
        if element_counts[subscript_part] == 2:
            contractLeg1 = openEdgeOrdering1[subscript1.index(subscript_part)]
            contractLeg2 = openEdgeOrdering2[subscript2.index(subscript_part)]

            contractLegList1.append(contractLeg1)
            contractLegList2.append(contractLeg2)

            matchIrrepsOnLegs.append((contractLeg1, contractLeg2))

    return matchIrrepsOnLegs, contractLegList1, contractLegList2


def einsum_get_openLegReplacements(su2tensor1, su2tensor2, contractLegList1, contractLegList2):
    """
    Get the new names of the open edges for the contracted tensors.
    """
    openLegReplacements1 = []
    openLegReplacements2 = []
    newListOfOpenEdges = []

    openEdgeOrdering1 = [edge for edge in su2tensor1.nameOrdering if edge < 0]
    openEdgeOrdering2 = [edge for edge in su2tensor2.nameOrdering if edge < 0]

    newOpenEdgename = -1
    newOpenEdgeNumber = 1
    # we name the remaining open legs such that we start going through tensor1 and then tensor 2
    for openEdge in openEdgeOrdering1:
        # we have an open leg that in not contracted
        if openEdge not in contractLegList1:
            openLegReplacements1.append((openEdge, newOpenEdgename))

            # update the original entry in listOfOpenEdges
            openEdgeEntry = [entry for entry in su2tensor1.listOfOpenEdges if entry['edgeName'] == openEdge]
            assert len(openEdgeEntry) == 1
            openEdgeEntry = openEdgeEntry[0].copy()
            openEdgeEntry['edgeName'], openEdgeEntry['edgeNumber'] = newOpenEdgename, newOpenEdgeNumber
            newListOfOpenEdges.append(openEdgeEntry)

            newOpenEdgename -= 1
            newOpenEdgeNumber += 1

    for openEdge in openEdgeOrdering2:
        # we have an open leg that in not contracted
        if openEdge not in contractLegList2:
            openLegReplacements2.append((openEdge, newOpenEdgename))

            # update the original entry in listOfOpenEdges
            openEdgeEntry = [entry for entry in su2tensor2.listOfOpenEdges if entry['edgeName'] == openEdge]
            assert len(openEdgeEntry) == 1
            openEdgeEntry = openEdgeEntry[0].copy()
            openEdgeEntry['edgeName'], openEdgeEntry['edgeNumber'] = newOpenEdgename, newOpenEdgeNumber
            newListOfOpenEdges.append(openEdgeEntry)

            newOpenEdgename -= 1
            newOpenEdgeNumber += 1

    return [openLegReplacements1, openLegReplacements2], newListOfOpenEdges


def einsum_getNewFusionTree(matchIrrepsOnLegs, openLegReplacements,
                            oldFusionTree1, oldFusionTreeDirection1,
                            oldFusionTree2, oldFusionTreeDirection2,
                            numberOfInternalEdges1, numberOfInternalEdges2):
    """
    Get the new fusion tree for the contracted tensor.
    Example of contractionList: [-3, -4], [-1, -2], -3 of 1 with -1 of 2 and -4 of 1 with -2 of 2.
    """
    matchingLegsIsOpen = [(0, 0) for _ in range(len(matchIrrepsOnLegs))]

    newFusionTree1 = oldFusionTree1.copy()
    newFusionTree2 = pd.DataFrame(oldFusionTree2.copy())
    # rename the internal edges of the second fusionTree
    newFusionTree2 = newFusionTree2.replace(
        range(1, numberOfInternalEdges2+1),
        range(numberOfInternalEdges1+1, numberOfInternalEdges1+numberOfInternalEdges2+1)).values.tolist()

    internalLegReplacements = [[(intEdge, intEdge) for intEdge in range(1, numberOfInternalEdges1+1)]]
    internalLegReplacements.append([(oirgIntEdge, newIntEdge) for oirgIntEdge, newIntEdge in
                                    zip(range(1, numberOfInternalEdges2+1),
                                        range(numberOfInternalEdges1+1, numberOfInternalEdges1+numberOfInternalEdges2+1)
                                        )
                                    ])

    # rename the contracted legs such that they fit together in the two fusion trees.
    fusionTree1_df = pd.DataFrame(newFusionTree1)
    fusionTree2_df = pd.DataFrame(newFusionTree2)
    replaceName = numberOfInternalEdges1 + numberOfInternalEdges2 + 1
    for idx, (contractLeg1, contractLeg2) in enumerate(matchIrrepsOnLegs):
        openLegReplacements[0].append((contractLeg1, replaceName))
        openLegReplacements[1].append((contractLeg2, replaceName))

        matchIrrepsOnLegs[idx] = (replaceName, replaceName)
        replaceName += 1

    # rename the edges such they match the new given names
    openLegReplacements1 = np.array(openLegReplacements[0]).transpose()
    openLegReplacements2 = np.array(openLegReplacements[1]).transpose()
    fusionTree1_df = fusionTree1_df.replace(openLegReplacements1[0], openLegReplacements1[1])
    fusionTree2_df = fusionTree2_df.replace(openLegReplacements2[0], openLegReplacements2[1])

    newFusionTree1 = fusionTree1_df.values.tolist()
    newFusionTree2 = fusionTree2_df.values.tolist()
    newFusionTree = newFusionTree1 + newFusionTree2
    newFusionTreeDirections = oldFusionTreeDirection1 + oldFusionTreeDirection2

    # irreps that need to be dropped
    dropIrreps = []
    # remove the loops
    foundLoop = True
    while foundLoop:
        foundLoop = False
        for idx1, (node1, direction1) in enumerate(zip(newFusionTree, newFusionTreeDirections)):
            # fusion node with two internal legs on the incoming legs
            if direction1 == -1 and node1[0] > 0 and node1[1] > 0:
                for idx2 in range(len(newFusionTree)):
                    if (newFusionTreeDirections[idx2] == +1) and ({node1[0], node1[1]} == {newFusionTree[idx2][1], newFusionTree[idx2][2]}):
                        # we have found a loop
                        foundLoop = True
                        combLeg2, combLeg1 = newFusionTree[idx1][2], newFusionTree[idx2][0]
                        matchIrrepsOnLegs.append((combLeg1, combLeg2))

                        dropIrreps.extend([node1[0], node1[1]])

                        # delete the two nodes (first larger idx)
                        newFusionTree.pop(max(idx1, idx2))
                        newFusionTreeDirections.pop(max(idx1, idx2))
                        newFusionTree.pop(min(idx1, idx2))
                        newFusionTreeDirections.pop(min(idx1, idx2))

                        if combLeg1 < 0 and combLeg2 < 0:
                            matchingLegsIsOpen.append((1, 1))
                            newFusionTree = [[0, combLeg1, combLeg2]]
                            newFusionTreeDirections = [-1]
                            break

                        elif combLeg2 < 0 and combLeg1 > 0:
                            combLeg1, combLeg2 = combLeg2, combLeg1
                            matchingLegsIsOpen.append((0, 1))

                        elif combLeg2 > 0 and combLeg1 < 0:
                            matchingLegsIsOpen.append((1, 0))

                        else:
                            matchingLegsIsOpen.append((0, 0))

                        # replace the name of combLeg2 with combLeg1
                        newFusionTree = (
                            pd.DataFrame(newFusionTree).replace([combLeg2], [combLeg1]).values.tolist()
                        )
                        dropIrreps.append(combLeg2)

                        # we don't have to search for another partner for node1; break the idx2 loop
                        break

    return (newFusionTree, newFusionTreeDirections, matchIrrepsOnLegs, matchingLegsIsOpen,
            internalLegReplacements, openLegReplacements, dropIrreps)


def einsum_newNameOrderingsForFusionTrees(oldNameOrdering1, oldNameOrdering2,
                                          openLegReplacements, internalLegReplacements):
    """
    Replace the names of the edges in the old nameOrderings.
    """
    openLegReplacements1 = np.array(openLegReplacements[0]).transpose()
    openLegReplacements2 = np.array(openLegReplacements[1]).transpose()
    internalLegReplacements1 = np.array(internalLegReplacements[0]).transpose()
    internalLegReplacements2 = np.array(internalLegReplacements[1]).transpose()

    newNameOrdering1 = pd.DataFrame(oldNameOrdering1).replace(openLegReplacements1[0], openLegReplacements1[1])
    if internalLegReplacements1.size != 0:
        newNameOrdering1 = newNameOrdering1.replace(internalLegReplacements1[0], internalLegReplacements1[1])

    newNameOrdering2 = pd.DataFrame(oldNameOrdering2).replace(openLegReplacements2[0], openLegReplacements2[1])
    if internalLegReplacements2.size != 0:
        newNameOrdering2 = newNameOrdering2.replace(internalLegReplacements2[0], internalLegReplacements2[1])

    return newNameOrdering1.values.flatten().tolist(), newNameOrdering2.values.flatten().tolist()


def einsum_getNewChargeSectors(oldChargeSectors1, oldChargeSectors2,
                               newNameOrdering1, newNameOrdering2,
                               matchIrrepsOnLegs,
                               oldListOfDegeneracyTensors1, oldListOfDegeneracyTensors2,
                               einsumOrdering, dropIrreps):
    """
    Get new charge sectors, degeneracy tensors and nameOrdering for the contracted tensor.
    """
    assert len(oldChargeSectors1) == len(oldListOfDegeneracyTensors1)
    assert len(oldChargeSectors2) == len(oldListOfDegeneracyTensors2)
    oldChargeSectors1 = pd.DataFrame(oldChargeSectors1, columns=newNameOrdering1)
    oldChargeSectors1['idx1'] = list(oldChargeSectors1.index)
    oldChargeSectors2 = pd.DataFrame(oldChargeSectors2, columns=newNameOrdering2)
    oldChargeSectors2['idx2'] = list(oldChargeSectors2.index)

    matchIrrepsOnLegs = np.array(matchIrrepsOnLegs).transpose()

    newChargeSectors = pd.merge(oldChargeSectors1, oldChargeSectors2,
                                left_on=list(matchIrrepsOnLegs[0]), right_on=list(matchIrrepsOnLegs[1]), how='inner')

    newChargeSectors = newChargeSectors.drop(columns=dropIrreps)

    newNameOrdering = list(newChargeSectors.columns)
    newNameOrdering.remove('idx1')
    newNameOrdering.remove('idx2')

    chargeSectorGroups = newChargeSectors.groupby(newNameOrdering)

    newChargeSectors, newDegeracyTensors = einsum_getNewDegeneracyTensors(chargeSectorGroups=chargeSectorGroups,
                                                                          oldListOfDegeneracyTensors1=oldListOfDegeneracyTensors1,
                                                                          oldListOfDegeneracyTensors2=oldListOfDegeneracyTensors2,
                                                                          einsumOrdering=einsumOrdering)

    newNameOrdering, newChargeSectors = change_order_of_nameOrdering(nameOrdering=newNameOrdering,
                                                                     chargeSectors=newChargeSectors)

    return newNameOrdering, newChargeSectors, newDegeracyTensors


def change_order_of_nameOrdering(nameOrdering, chargeSectors):
    """
    Get the new name ordering and charge sectors after the contraction.
    """
    chargeSectors_df = pd.DataFrame(chargeSectors, columns=nameOrdering)
    internalLegs = [irrep for irrep in nameOrdering if irrep > 0]
    openLegs = [irrep for irrep in nameOrdering if irrep < 0]
    newNameOrdering = internalLegs + openLegs

    chargeSectors_df = chargeSectors_df[newNameOrdering]

    return newNameOrdering, chargeSectors_df.values.tolist()


def einsum_getNewDegeneracyTensors(chargeSectorGroups,
                                   oldListOfDegeneracyTensors1, oldListOfDegeneracyTensors2,
                                   einsumOrdering):
    """
    Helper function of einsum_getNewChargeSectors. Return the new degeneracy tensors and charge sectors for the
    contracted tensor.
    """
    newChargeSectors = []
    newDegeneracyTensors = []

    for chargeSector, df in chargeSectorGroups:
        newChargeSectors.append(list(chargeSector))

        idx1 = int(df.iloc[0]['idx1'])
        idx2 = int(df.iloc[0]['idx2'])
        mockDegTensor = np.einsum(oldListOfDegeneracyTensors1[idx1], einsumOrdering[0],
                                  oldListOfDegeneracyTensors2[idx2], einsumOrdering[1],
                                  einsumOrdering[2])
        newDegeneracyTensor = np.zeros(mockDegTensor.shape, dtype='complex128')

        for idx, row in df.iterrows():
            idx1 = int(row['idx1'])
            idx2 = int(row['idx2'])
            newDegeneracyTensor += np.einsum(oldListOfDegeneracyTensors1[idx1], einsumOrdering[0],
                                             oldListOfDegeneracyTensors2[idx2], einsumOrdering[1],
                                             einsumOrdering[2])

        newDegeneracyTensors.append(newDegeneracyTensor)

    return newChargeSectors, newDegeneracyTensors


def einsum_makeinternalEdgesLabelingCorrect(newFusionTree, newNameOrdering):
    """
    Make sure the labels of the internal edges start with 1 and end with numberOfInternalEdges.
    """
    internalEdgesLabels = [intEdge for intEdge in newNameOrdering if intEdge > 0]
    if sorted(internalEdgesLabels) == sorted(list(range(1, len(internalEdgesLabels)+1))):
        return newFusionTree, newNameOrdering

    wrongLabels = [intEdge for intEdge in internalEdgesLabels
                   if intEdge not in list(range(1, len(internalEdgesLabels) + 1))]

    replaceBy = [newLabel for newLabel in list(range(1, len(internalEdgesLabels) + 1))
                 if newLabel not in internalEdgesLabels]

    newNameOrdering = pd.DataFrame(newNameOrdering).replace(wrongLabels, replaceBy)
    newNameOrdering = newNameOrdering.values.flatten().tolist()

    newFusionTree = pd.DataFrame(newFusionTree).replace(wrongLabels, replaceBy)
    newFusionTree = newFusionTree.values.tolist()

    return newFusionTree, newNameOrdering

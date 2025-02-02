import numpy as np
import pandas as pd


def find_unnecessary_nodes(fusionTree, fusionTreeDirections):
    # if we only have one node, then there can't be any unnecessary nodes.
    if len(fusionTree) == 1:
        return []

    unnecessaryNodes = []
    for node, direction in zip(fusionTree, fusionTreeDirections):
        if direction == -1:
            if 0 in [node[0], node[1]]:
                unnecessaryNodes.append((node, direction))

        elif direction == +1:
            if 0 in [node[1], node[2]]:
                unnecessaryNodes.append((node, direction))

    return unnecessaryNodes


def update_node(fusionTree, toReplace_leg, replacing_leg):
    # node that needs to be updated
    updateNode = [node for node in fusionTree if toReplace_leg in node][0]

    # remove the node
    updateNode_idx = fusionTree.index(updateNode)

    updateEntry_idx = updateNode.index(toReplace_leg)
    updateNode[updateEntry_idx] = replacing_leg

    # update the node in the fusionTree
    fusionTree[updateNode_idx] = updateNode

    return fusionTree


def update_chargeSectors(chargeSectors, nameOrdering, fusionTree, toReplace_leg):
    chargeSector_df = pd.DataFrame(chargeSectors, columns=nameOrdering)
    chargeSector_df = chargeSector_df.drop(toReplace_leg, axis=1)
    chargeSectors = chargeSector_df.values.tolist()

    nameOrdering.remove(toReplace_leg)
    # toReplace_leg is an internal leg.
    # We have to rename the internal legs, suh that they count from 1 to numberInternalLegs
    internalEdges = [irrep for irrep in nameOrdering if irrep > 0]

    df_tree = pd.DataFrame(fusionTree)
    # rename the internal edges
    fusionTree = df_tree.replace(internalEdges, range(1, len(internalEdges)+1)).values.tolist()
    for idx, internalLegName in enumerate(range(1, len(internalEdges)+1)):
        nameOrdering[idx] = internalLegName

    return fusionTree, nameOrdering, chargeSectors


def remove_unnecessary_nodes(chargeSectors, nameOrdering, fusionTree, fusionTreeDirections):
    unnecessaryNodes = find_unnecessary_nodes(fusionTree, fusionTreeDirections)

    for node, direction in unnecessaryNodes:
        node_idx = fusionTree.index(node)
        fusionTree.pop(node_idx)
        fusionTreeDirections.pop(node_idx)
        if direction == -1:
            toReplace_leg = node[2]
            replacing_leg = [leg for leg in node if leg not in [0, node[2]]][0]

        elif direction == +1:
            toReplace_leg = node[0]
            replacing_leg = [leg for leg in node if leg not in [0, node[0]]][0]

        fusionTree = update_node(fusionTree, toReplace_leg, replacing_leg)

        fusionTree, nameOrdering, chargeSectors = update_chargeSectors(chargeSectors, nameOrdering,
                                                                       fusionTree, toReplace_leg)

    return fusionTree, fusionTreeDirections, nameOrdering, chargeSectors


if __name__ == "__main__":
    from su2tn.su2_tensor import SU2Tensor
    fusionTree = [[-1, -2, 1], [1, 0, -3]]
    fusionTreeDirections = [-1, +1]

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
        listOfDegeneracyTensors.append(np.random.rand(2, 2, 2, 2))
    su2tensor.listOfDegeneracyTensors = listOfDegeneracyTensors

    fusionTree, fusionTreeDirections, nameOrdering, chargeSectors = remove_unnecessary_nodes(su2tensor.listOfChargeSectors, su2tensor.nameOrdering, su2tensor.fusionTree, su2tensor.fusionTreeDirections)

    print(fusionTree)
    print(fusionTreeDirections)

    print(nameOrdering)
    print(chargeSectors)



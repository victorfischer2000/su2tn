import pandas as pd


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
    print(chargeSector_df)
    chargeSector_df = chargeSector_df.drop(toReplace_leg, axis=1)
    print(chargeSector_df)
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

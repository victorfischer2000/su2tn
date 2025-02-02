from su2tn.su2tensor_utils.remove_unnec_dummylegs import update_node, update_chargeSectors, remove_unnecessary_nodes


def find_unnecessary_nodes_twoDummies(fusionTree, fusionTreeDirections):
    """
    Find all nodes with two dummy indices
    """
    # if we only have one node, then there can't be any unnecessary nodes.
    if len(fusionTree) == 1:
        return []

    unnecessaryNodes = []
    for node, direction in zip(fusionTree, fusionTreeDirections):
        if node.count(0) == 2:
            unnecessaryNodes.append((node, direction))

    return unnecessaryNodes


def remove_unnecessary_nodes_twoDummies(chargeSectors, nameOrdering, fusionTree, fusionTreeDirections):
    unnecessaryNodes = find_unnecessary_nodes_twoDummies(fusionTree, fusionTreeDirections)
    """
    Function that updates the fusion tree, nameOrdering and charge sectors when removing unnecessary nodes.
    """

    for node, direction in unnecessaryNodes:
        node_idx = fusionTree.index(node)
        fusionTree.pop(node_idx)
        fusionTreeDirections.pop(node_idx)

        toReplace_leg = [leg for leg in node if leg != 0]
        assert len(toReplace_leg) == 1
        toReplace_leg = toReplace_leg[0]

        replacing_leg = 0

        fusionTree = update_node(fusionTree, toReplace_leg, replacing_leg)

        fusionTree, nameOrdering, chargeSectors = update_chargeSectors(chargeSectors, nameOrdering,
                                                                       fusionTree, toReplace_leg)

    return fusionTree, fusionTreeDirections, nameOrdering, chargeSectors

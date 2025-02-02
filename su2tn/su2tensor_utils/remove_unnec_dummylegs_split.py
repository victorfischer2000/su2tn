import numpy as np
import pandas as pd
from su2tn.su2tensor_utils.remove_unnec_dummylegs import update_node, update_chargeSectors, remove_unnecessary_nodes


def find_unnecessary_nodes_split(fusionTree, fusionTreeDirections):
    """
    Find all nodes with two dummy indices
    """
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


def remove_unnecessary_nodes_split(chargeSectors, nameOrdering, fusionTree, fusionTreeDirections):
    unnecessaryNodes = find_unnecessary_nodes_split(fusionTree, fusionTreeDirections)

    for node, direction in unnecessaryNodes:
        node_idx = fusionTree.index(node)
        fusionTree.pop(node_idx)
        fusionTreeDirections.pop(node_idx)
        if direction == -1:
            # get the internal leg. If two internal legs then just get the first on
            toReplace_leg = [leg for leg in node if leg > 0][0]
            replacing_leg = [leg for leg in node if leg not in [0, toReplace_leg]][0]

        elif direction == +1:
            toReplace_leg = [leg for leg in node if leg > 0][0]
            replacing_leg = [leg for leg in node if leg not in [0, toReplace_leg]][0]

        fusionTree = update_node(fusionTree, toReplace_leg, replacing_leg)

        fusionTree, nameOrdering, chargeSectors = update_chargeSectors(chargeSectors, nameOrdering,
                                                                       fusionTree, toReplace_leg)

    return fusionTree, fusionTreeDirections, nameOrdering, chargeSectors

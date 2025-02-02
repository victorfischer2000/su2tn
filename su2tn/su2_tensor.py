import pandas as pd
from su2tn.util import *
from su2tn.su2tensor_utils.calc_charge_sectors_util import calculate_all_charge_sectors
from su2tn.su2tensor_utils.f_move_util import f_move_perform_fmove
from su2tn.su2tensor_utils.switch_to_given_fusion_tree_util import f_move_switch_fusion_tree
from su2tn.su2tensor_utils.contraction_util import einsum
from su2tn.su2tensor_utils.permutation_util import transpose
from su2tn.su2tensor_utils.fuse_neighborLegs import fuse_neighboring_legs
from su2tn.su2tensor_utils.split_Leg_util import split_leg
from su2tn.su2tensor_utils.permutation_util import perform_permute
from su2tn.su2tensor_utils.reversing_util import reverse_right_most_leg
from su2tn.su2tensor_utils.reverse_left_leg import reverse_left_most_leg


class SU2Tensor:

    def __init__(self, listOfOpenEdges, listOfDegeneracyTensors, fusionTree, fusionTreeDirections,
                 listOfChargeSectors=None, nameOrdering=None):
        self.numberOfOpenEdges = len(listOfOpenEdges)
        self.numberOfInternalEdges = len(fusionTree) - 1 # len(listOfOpenEdges) - 3
        self.numberOfAuxiliaryEdges = None
        self.listOfOpenEdges = listOfOpenEdges
        self.listOfInternalEdges = None
        # TODO: Generate nameOrdering from listOFOpenEdges
        if nameOrdering is None:
            self.nameOrdering = [l for l in range(1, self.numberOfInternalEdges + 1)] +\
                                [-k for k in range(1, self.numberOfOpenEdges + 1)]
        else:
            self.nameOrdering = nameOrdering
        self.listOfDegeneracyTensors = listOfDegeneracyTensors
        self.listOfStructuralTensors = None
        self.fusionTree = fusionTree
        self.fusionTreeDirections = fusionTreeDirections
        if listOfChargeSectors is None:
            self.listOfChargeSectors = self.calculate_all_charge_sectors()
        else:
            self.listOfChargeSectors = listOfChargeSectors

    def calculate_all_charge_sectors(self):
        """
        From the list of open edges and the structure of the fusionTree, calculate all possible charge sectors.
        """

        charge_sectors = calculate_all_charge_sectors(listOfOpenEdges=self.listOfOpenEdges,
                                                      nameOrdering=self.nameOrdering,
                                                      fusionTree=self.fusionTree,
                                                      fusionTreeDirections=self.fusionTreeDirections)

        return charge_sectors

    def perform_f_move(self, fNodes):
        """
        Perform an F-Move on the two nodes given by fNodes.
        """
        # calls the function from t_move_util module, that changes the su2-tensor in-place.
        f_move_perform_fmove(su2tensor=self, fNodes=fNodes)

    def transpose(self, axes):
        """
        Transpose tensor. Only works between all incoming or all outgoing legs.
        """
        transpose(su2tensor=self, axes=axes)

    def fuse_neighboring_legs(self, irrepName1, irrepName2):
        """

        """
        fuse_neighboring_legs(su2tensor=self, irrep1Name=irrepName1, irrep2Name=irrepName2)

    def split_leg(self, splitIrrepName):
        """

        """
        split_leg(su2tensor=self, splitLegName=splitIrrepName)

    def perform_permutation(self, irrep1Name, irrep2Name):
        perform_permute(su2tensor=self, irrep1=irrep1Name, irrep2=irrep2Name)

    def reverse_right_most_leg(self, reverseIrrepName):
        reverse_right_most_leg(su2tensor=self, reverseLeg=reverseIrrepName)

    def reverse_left_most_leg(self, reverseIrrepName):
        reverse_left_most_leg(su2tensor=self, reverseLeg=reverseIrrepName)

    def switch_fusion_tree(self, newFusionTree, newFusionTreeDirections):
        """
        Switches the su2-tensor to the given fusion tree. The new fusion tree can have a different labeling of internal
        edges, only the structure matters.
        """
        f_move_switch_fusion_tree(su2tensor=self,
                                  newFusionTree=newFusionTree, newFusionTreeDirections=newFusionTreeDirections)

    def return_explicit_tensor_blocks(self):
        """
        Returns the explicit tensors for each of the charge combinations of the outer legs as a dict.
        """
        chargeBlockDict = {}
        cs_df = pd.DataFrame(self.listOfChargeSectors, columns=self.nameOrdering)
        cs_groups = cs_df.groupby([edge for edge in self.nameOrdering if edge < 0])
        # we have a block for each combination of irreps in outer legs.
        for outerIrreps, relevantCS in cs_groups:
            # build dummy tensor
            actualTensor = np.kron(
                np.zeros(self.listOfDegeneracyTensors[relevantCS.index[0]].shape),
                np.zeros(self.build_structural_tensor(charge_sector=relevantCS.values.tolist()[0],
                                                      nameOrdering=self.nameOrdering,
                                                      fusionTree=self.fusionTree,
                                                      fusionTreeDirections=self.fusionTreeDirections).shape).astype('complex128')
            )

            # calculate actual tensor for the block, defined by the irrep-combination of the outer legs.
            for idx, row in relevantCS.iterrows():
                cs = list(row.array)
                actualTensor += np.kron(self.listOfDegeneracyTensors[idx],
                                        self.build_structural_tensor(charge_sector=cs,
                                                                     nameOrdering=self.nameOrdering,
                                                                     fusionTree=self.fusionTree,
                                                                     fusionTreeDirections=self.fusionTreeDirections))

            chargeBlockDict[outerIrreps] = actualTensor

        return chargeBlockDict

    @classmethod
    def build_structural_tensor(cls, charge_sector, nameOrdering, fusionTree, fusionTreeDirections):
        """
        Returns the explicit form a fusion tree tensor for a given charge sector.
        """
        charge_sector = [0] + charge_sector.copy()
        nameOrdering = [0] + nameOrdering.copy()

        fusionTree = fusionTree.copy()
        fusionTreeDirections = fusionTreeDirections.copy()

        fusionTreeNode = fusionTree.pop(0)
        fusionTreeDirection = fusionTreeDirections.pop(0)
        names_of_open_legs = [fusionTreeNode[0], fusionTreeNode[1], fusionTreeNode[2]]
        if fusionTreeDirection == -1:
            t = Cfuse(charge_sector[nameOrdering.index(fusionTreeNode[0])],
                      charge_sector[nameOrdering.index(fusionTreeNode[1])],
                      charge_sector[nameOrdering.index(fusionTreeNode[2])])
        else:
            t = Csplit(charge_sector[nameOrdering.index(fusionTreeNode[0])],
                       charge_sector[nameOrdering.index(fusionTreeNode[1])],
                       charge_sector[nameOrdering.index(fusionTreeNode[2])])

        while fusionTree:
            # find a node that has a common leg with output tensor
            for fusionTreeNode in fusionTree:
                if any(leg in names_of_open_legs for leg in fusionTreeNode):
                    contract_tensor_idx = fusionTree.index(fusionTreeNode)
                    contract_tensor_legs = fusionTree.pop(contract_tensor_idx)
                    contract_tensor_direction = fusionTreeDirections.pop(contract_tensor_idx)
                    break

            # create the split or fuse tensor for the node that will be contracted to the output tensor
            if contract_tensor_direction == -1:
                contract_tensor = Cfuse(charge_sector[nameOrdering.index(contract_tensor_legs[0])],
                                        charge_sector[nameOrdering.index(contract_tensor_legs[1])],
                                        charge_sector[nameOrdering.index(contract_tensor_legs[2])])
            else:
                contract_tensor = Csplit(charge_sector[nameOrdering.index(contract_tensor_legs[0])],
                                         charge_sector[nameOrdering.index(contract_tensor_legs[1])],
                                         charge_sector[nameOrdering.index(contract_tensor_legs[2])])

            # get the dimensions over which to contract
            common_legs = list(set(names_of_open_legs) & set(contract_tensor_legs))
            contr_dimension_t = []
            contr_dimension_contract_tensor = []
            for common_leg in common_legs:
                contr_dimension_t.append(names_of_open_legs.index(common_leg))
                contr_dimension_contract_tensor.append(contract_tensor_legs.index(common_leg))

            # do the contraction
            t = np.tensordot(t, contract_tensor, axes=(contr_dimension_t, contr_dimension_contract_tensor))

            # remove the common legs from open legs list and add the new open legs
            names_of_open_legs = [leg for leg in names_of_open_legs if leg not in common_legs]
            names_of_open_legs = names_of_open_legs + [leg for leg in contract_tensor_legs if leg not in common_legs]

        # permute the output-tensor indices to the correct permutation
        # open edges are labeled in the order 0, -1, -2, -3, -4,...

        perm = list(reversed(np.argsort(names_of_open_legs)))
        t = np.transpose(t, perm)
        if 0 in names_of_open_legs:
            # remove the dummy dimension
            t = np.reshape(t, list(t.shape)[1:])

        openEdgeOrdering = [edge for edge in nameOrdering if edge < 0]
        perm = list(reversed(np.argsort(openEdgeOrdering)))
        t = np.transpose(t, perm)

        return t

    @classmethod
    def einsum(cls, su2tensor1, subscript1, su2tensor2, subscript2):
        (newListOfOpenEdges, newNameOrdering, newChargeSectors, newDegeracyTensors,
         newFusionTree, newFusionTreeDirections) = einsum(su2tensor1=su2tensor1, subscript1=subscript1,
                                                          su2tensor2=su2tensor2, subscript2=subscript2)

        return SU2Tensor(listOfOpenEdges=newListOfOpenEdges,
                         listOfDegeneracyTensors=newDegeracyTensors,
                         fusionTree=newFusionTree,
                         fusionTreeDirections=newFusionTreeDirections,
                         listOfChargeSectors=newChargeSectors,
                         nameOrdering=newNameOrdering)


if __name__ == '__main__':
    fusionTree = [[0, -1, 1], [1, -2, -3]]
    fusionTreeDirections = [+1, +1]

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

    # calculate the explicit tensors
    control_dict = su2tensor.return_explicit_tensor_blocks()

    # do the f-move
    su2tensor.perform_f_move([[0, -1, 1], [1, -2, -3]])

    testFusionTree = su2tensor.fusionTree.copy()
    controlFusionTree = [[0, 1, -3], [1, -1, -2]]
    assert sorted(testFusionTree) == sorted(controlFusionTree)

    # calculate the explicit tensors
    test_dict = su2tensor.return_explicit_tensor_blocks()
    for outerIrreps in test_dict.keys():
        assert np.allclose(control_dict[outerIrreps], test_dict[outerIrreps])

    # do the f-move
    su2tensor.perform_f_move([[0, 1, -3], [1, -1, -2]])

    # calculate the explicit tensors
    test_dict = su2tensor.return_explicit_tensor_blocks()
    for outerIrreps in test_dict.keys():
        assert np.allclose(control_dict[outerIrreps], test_dict[outerIrreps])
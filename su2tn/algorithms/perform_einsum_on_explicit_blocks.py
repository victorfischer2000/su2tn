import numpy as np


def perform_einsum_on_explicit_tensor_blocks(A1, subscript1, A2, subscript2):
    from su2tn.su2tensor_utils.contraction_util import einsum_get_einsumOrdering
    einsumOrdering, element_counts = einsum_get_einsumOrdering(subscript1, subscript2)

    subscript1 = list(subscript1)
    subscript2 = list(subscript2)

    contract_subscript = []
    for subscript_part in element_counts:
        if element_counts[subscript_part] == 2:
            contract_subscript.append(subscript_part)

    contract_idxs1 = []
    contract_idxs2 = []
    for contract in contract_subscript:
        contract_idxs1.append(subscript1.index(contract))
        contract_idxs2.append(subscript2.index(contract))

    # dictionary, where all the tensors to the charge sectors are stored
    contracted_dict = {}

    for cs1, deg1 in A1.items():
        for cs2, deg2 in A2.items():
            chargeSector1 = list(cs1).copy()
            chargeSector2 = list(cs2).copy()
            # check if all the irreps are matching
            if all(chargeSector1[subscript1.index(contract)] == chargeSector2[subscript2.index(contract)]
                   for contract in contract_subscript):
                # do the contraction
                out = np.einsum(deg1, subscript1, deg2, subscript2)

                # get new cs
                for contract_idx1 in reversed(sorted(contract_idxs1)):
                    chargeSector1.pop(contract_idx1)

                for contract_idx2 in reversed(sorted(contract_idxs2)):
                    chargeSector2.pop(contract_idx2)

                newcs = tuple(chargeSector1 + chargeSector2)

                # add if a contraction already mapped to the same charge sector
                if newcs in contracted_dict.keys():
                    contracted_dict[newcs] += out
                else:
                    contracted_dict[newcs] = out

    return contracted_dict

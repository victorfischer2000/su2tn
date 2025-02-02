import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
import warnings
from su2tn.su2_tensor import SU2Tensor
from su2tn.algorithms.scalar_product import norm_of_single_fusionNode_tensor, vdot_of_two_single_fusionNodeode_tensors


def lanczos_step_apply_function(A:SU2Tensor, H:SU2Tensor, L:SU2Tensor, R:SU2Tensor):
    output = SU2Tensor.einsum(su2tensor1=L, subscript1=(0, 1, 2), su2tensor2=H, subscript2=(3, 1, 4, 5))
    output.perform_f_move([[-1, 2, -2], [-3, 2, 1]])

    output.perform_f_move([node for node in output.fusionTree if 1 in node])

    output = output.einsum(su2tensor1=output, subscript1=(0, 1, 2, 3, 4), su2tensor2=A, subscript2=(4, 1, 5))

    output = output.einsum(su2tensor1=output, subscript1=(0, 1, 2, 3), su2tensor2=R, subscript2=(2, 3, 4))

    switch_names_of_fusion_legs(output)

    return output


def switch_names_of_fusion_legs(fusionNodeTensor):
    assert fusionNodeTensor.fusionTreeDirections == [-1]

    fusionNodeTensor.fusionTree = [[-1, -2, -3]]
    chargeSectors = pd.DataFrame(fusionNodeTensor.listOfChargeSectors, columns=fusionNodeTensor.nameOrdering)
    fusionNodeTensor.listOfChargeSectors = chargeSectors[[-2, -1, -3]].values.tolist()
    fusionNodeTensor.nameOrdering = [-1, -2, -3]

    assert fusionNodeTensor.listOfOpenEdges[0]['edgeName'] == -1
    assert fusionNodeTensor.listOfOpenEdges[1]['edgeName'] == -2

    fusionNodeTensor.listOfOpenEdges[0]['edgeName'] = -2
    fusionNodeTensor.listOfOpenEdges[0]['edgeNumber'] = 2

    fusionNodeTensor.listOfOpenEdges[1]['edgeName'] = -1
    fusionNodeTensor.listOfOpenEdges[1]['edgeNumber'] = 1

    for idx in range(len(fusionNodeTensor.listOfDegeneracyTensors)):
        fusionNodeTensor.listOfDegeneracyTensors[idx] = np.transpose(fusionNodeTensor.listOfDegeneracyTensors[idx], (1, 0, 2))

    # degeneracy tensors are already correctly transposed
    # fusionTreeDirections is correct


def lanczos_iteration(vstart:SU2Tensor, numiter:int, H:SU2Tensor, L:SU2Tensor, R:SU2Tensor):
    """
    Perform a "matrix free" Lanczos iteration.

    Args:
        Afunc:      "matrix free" linear transformation of a given vector
        vstart:     starting vector for iteration
        numiter:    number of iterations (should be much smaller than dimension of vstart)

    Returns:
        tuple: tuple containing
          - alpha:      diagonal real entries of Hessenberg matrix
          - beta:       off-diagonal real entries of Hessenberg matrix
          - V:          `len(vstart) x numiter` matrix containing the orthonormal Lanczos vectors
    """
    verify_laczos_step = False

    # normalize starting vector
    nrmv = norm_of_single_fusionNode_tensor(su2tensor=vstart)
    assert nrmv > 0
    # vstart = vstart / nrmv
    for idx, degTensor in enumerate(vstart.listOfDegeneracyTensors):
        vstart.listOfDegeneracyTensors[idx] = vstart.listOfDegeneracyTensors[idx] / nrmv

    alpha = np.zeros(numiter)
    beta = np.zeros(numiter-1)

    V = []
    V.append(vstart)

    for j in range(numiter-1):
        w = lanczos_step_apply_function(A=V[j], H=H, L=L, R=R)

        if verify_laczos_step:
            # print(V[j].listOfChargeSectors)
            w_verify = lanczos_step_apply_function_verify(A=V[j], H=H, L=L, R=R)
            w_test = w.return_explicit_tensor_blocks()

            # if an key is not in the test key list, then the corresponding tensor has to be zero
            delete_keys = []
            for key in w_verify.keys():
                if key not in w_test.keys():
                    assert np.allclose(w_verify[key], np.zeros(w_verify[key].shape))
                    delete_keys.append(key)

            for key in delete_keys:
                del w_verify[key]

            assert sorted(list(w_verify.keys())) == sorted(list(w_test.keys()))
            for cs in w_verify.keys():
                assert np.allclose(w_verify[cs], w_test[cs])


        alphaj = vdot_of_two_single_fusionNodeode_tensors(w, V[j])
        assert np.isclose(alphaj.imag, 0, atol=1e-10)
        alpha[j] = alphaj.real

        # w -= alpha[j]*V[j] + (beta[j-1]*V[j-1] if j > 0 else 0)
        for chargeSector in w.listOfChargeSectors:
            w.listOfDegeneracyTensors[w.listOfChargeSectors.index(chargeSector)] -= (
                alpha[j] * V[j].listOfDegeneracyTensors[V[j].listOfChargeSectors.index(chargeSector)])

            if j != 0:
                w.listOfDegeneracyTensors[w.listOfChargeSectors.index(chargeSector)] -= (
                        beta[j-1] * V[j-1].listOfDegeneracyTensors[V[j-1].listOfChargeSectors.index(chargeSector)])

        betaj = norm_of_single_fusionNode_tensor(w)
        assert np.isclose(betaj.imag, 0, atol=1e-6)
        beta[j] = betaj.real
        if np.isclose(beta[j], 0, atol=1e-6):
            warnings.warn(
                f'beta[{j}] ~= 0 encountered during Lanczos iteration.',
                RuntimeWarning)
            # premature end of iteration
            numiter = j + 1
            return (alpha[:numiter], beta[:numiter-1], V[:numiter])

        # w = w / beta[j]
        for chargeSector in w.listOfChargeSectors:
            w.listOfDegeneracyTensors[w.listOfChargeSectors.index(chargeSector)] = (
                        w.listOfDegeneracyTensors[w.listOfChargeSectors.index(chargeSector)] / beta[j])

        V.append(w)

    # complete final iteration
    j = numiter-1
    w = lanczos_step_apply_function(A=V[j], H=H, L=L, R=R)
    alphaj = vdot_of_two_single_fusionNodeode_tensors(w, V[j])
    assert np.isclose(alphaj.imag, 0, atol=1e-10)
    alpha[j] = alphaj.real
    return (alpha, beta, V)


def eigh_krylov(vstart:SU2Tensor, numiter:int, H:SU2Tensor, L:SU2Tensor, R:SU2Tensor):
    """
    Compute Krylov subspace approximation of eigenvalues and vectors.
    """
    alpha, beta, V = lanczos_iteration(vstart=vstart, numiter=numiter, H=H, L=L, R=R)
    # diagonalize Hessenberg matrix
    w_hess, u_hess = eigh_tridiagonal(alpha, beta)
    print('Eigenwerte: ', w_hess)
    minEig = w_hess[0]
    u_min = u_hess[:, 0]
    # minEig = w_hess[-1]
    # u_min = u_hess[:, -1]

    out = V[0]

    for idx, degTensor in enumerate(out.listOfDegeneracyTensors):
        out.listOfDegeneracyTensors[idx] = out.listOfDegeneracyTensors[idx] * u_min[0]

    # compute Ritz eigenvectors
    for i in range(1, len(u_min)):
        # out += u_min[i] * V[i]
        for chargeSector in V[i].listOfChargeSectors:
            # TODO: Check if this is necessary because the charge sectors should not change for the different V[i]
            if chargeSector not in out.listOfChargeSectors:
                out.listOfChargeSectors.append(chargeSector)

                # out.listOfDegeneracyTensors.append(out.listOfDegeneracyTensors[V[i].listOfChargeSectors.index(chargeSector)])
                out.listOfDegeneracyTensors.append(
                    u_min[i] * V[i].listOfDegeneracyTensors[V[i].listOfChargeSectors.index(chargeSector)])
            else:
                out.listOfDegeneracyTensors[out.listOfChargeSectors.index(chargeSector)] += (
                    u_min[i] * V[i].listOfDegeneracyTensors[V[i].listOfChargeSectors.index(chargeSector)])

    return (minEig, out)


def lanczos_step_apply_function_verify(A:SU2Tensor, H:SU2Tensor, L:SU2Tensor, R:SU2Tensor):
    L_explicit = L.return_explicit_tensor_blocks()
    H_explicit = H.return_explicit_tensor_blocks()
    A_explicit = A.return_explicit_tensor_blocks()
    R_explicit = R.return_explicit_tensor_blocks()

    output = perform_einsum_on_explicit_tensor_blocks(A1=L_explicit, subscript1=(0, 1, 2),
                                                      A2=H_explicit, subscript2=(3, 1, 4, 5))

    output = perform_einsum_on_explicit_tensor_blocks(A1=output, subscript1=(0, 1, 2, 3, 4),
                                                      A2=A_explicit, subscript2=(4, 1, 5))

    output = perform_einsum_on_explicit_tensor_blocks(A1=output, subscript1=(0, 1, 2, 3),
                                                      A2=R_explicit, subscript2=(2, 3, 4))

    transposed_output = {}
    for out_cs, out_deg in output.items():
        out_cs = list(out_cs)
        assert len(out_cs) == 3
        out_cs = tuple([out_cs[1], out_cs[0], out_cs[2]])
        out_deg = np.transpose(out_deg, (1, 0, 2))

        transposed_output[out_cs] = out_deg

    return transposed_output


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


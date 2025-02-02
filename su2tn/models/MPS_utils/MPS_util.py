import numpy as np

def get_full_su2tensor_from_MPS_list(MPS):
    MPS_list = MPS.A
    output = MPS_list[0]
    for i in range(1, len(MPS_list)):
        openEdgeOrder = [j for j in output.nameOrdering if j < 0]
        output = output.einsum(su2tensor1=output, subscript1=tuple([i for i in range(len(openEdgeOrder))]),
                               su2tensor2=MPS_list[i], subscript2=(len(openEdgeOrder), len(openEdgeOrder) - 1, len(openEdgeOrder) + 1))

    return output


def get_explizit_tensor_state(MPS):
    MPSlist = MPS.A

    left_MPS = MPSlist[0].return_explicit_tensor_blocks()

    new_tensor_dict = {}
    for key, tensor in left_MPS.items():
        new_tensor_dict[list(key)[2]] = tensor

    for MPS in MPSlist[1:]:
        old_tensor_dict = new_tensor_dict
        new_tensor_dict = {}
        explicitMPS = MPS.return_explicit_tensor_blocks()

        for key, tensor in explicitMPS.items():
            if list(key)[1] in old_tensor_dict.keys():
                new_tensor = np.tensordot(
                    old_tensor_dict[list(key)[1]],
                    tensor,
                    (len(old_tensor_dict[list(key)[1]].shape)-1, 1)
                )

                if list(key)[2] in new_tensor_dict.keys():
                    new_tensor_dict[list(key)[2]] += new_tensor
                else:
                    new_tensor_dict[list(key)[2]] = new_tensor

    assert len(list(new_tensor_dict.keys())) == 1

    return list(new_tensor_dict.values())[0]
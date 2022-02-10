from typing import Dict, Any

config: Dict[str, Any] = {
    # Maximal number of input/output tensors for variadic operators
    'spec.max_num': 4,
    # Minimal rank of tensor (1/2)
    # 1: any tensor; 2: tensor for deep learning with the first dimension as batch size
    'spec.min_rank': 1,
    # Maximal rank of tensor
    'spec.max_rank': 5,
    # Maximal dimension value in tensor shape
    'spec.max_dim': 128,
    # Lower bound of small float
    'spec.min_small_float': 1e-5,
    # Upper bound of small float
    'spec.max_small_float': 1e-3,

    # Maximal number of model candidates
    'solver.max_model_cand': 4,
    # Length (in bits) of bit vector
    'solver.bit_vec_len': 32,

    # Maximal kernel size of convolution
    'op.max_kernel': 7,
    # Maximal stride of convolution
    'op.max_stride': 3,
    # Maximal padding
    'op.max_padding': 8,
    # Maximal dilation rate of convolution
    'op.max_dilation': 3,
}

from typing import Dict, Any

config: Dict[str, Any] = {
    # Maximal number of input/output tensors for variadic operators
    'spec.max_num': 4,
    # Minimal rank of tensor,
    'spec.min_rank': 1,
    # Maximal rank of tensor
    'spec.max_rank': 5,
    # Maximal dimension value in tensor shape
    'spec.max_dim': 128,

    # Maximal number of model candidates
    'solver.max_model_cand': 4,

    # Maximal kernel size of convolution
    'op.max_kernel': 7,
    # Maximal stride of convolution
    'op.max_stride': 3,
    # Maximal padding
    'op.max_padding': 8,
    # Maximal dilation rate of convolution
    'op.max_dilation': 3,
}

from typing import Dict, Any

config: Dict[str, Any] = {
    # Maximal number of input/output tensors for variadic operators
    'spec.max_num': 4,
    # Maximal rank of tensor
    'spec.max_rank': 5,
    # Maximal dimension value in tensor shape
    'spec.max_dim': 64,

    # Maximal number of model candidates
    'solver.max_model_cand': 4,
    # Length (in bits) of bit vector
    'solver.bit_vec_len': 32,

    # Maximal number of operation vertices in a graph
    'graph.max_opr_num': 50,
    # Penalty coefficient on number of uses of a value
    'graph.use_penal': 1,
    # Number of trials for generating one operation
    # For variadic operators, this is the maximal number of trials of adding a new input value
    'graph.opr_trials': 3,

    # Maximal kernel size of convolution
    'op.max_kernel': 7,
    # Maximal stride of convolution
    'op.max_stride': 2,
    # Maximal padding
    'op.max_padding': 6,
    # Maximal dilation rate of convolution
    'op.max_dilation': 2,
}

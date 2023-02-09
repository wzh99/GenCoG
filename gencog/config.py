from typing import Dict, Any

params: Dict[str, Any] = {
    # Maximal number of input tensors for variadic operators
    'spec.max_in_num': 4,
    # Maximal number of output tensors for variadic operators
    'spec.max_out_num': 3,
    # Maximal rank of tensor
    'spec.max_rank': 5,
    # Maximal dimension value in tensor shape
    'spec.max_dim': 4,

    # Maximal number of model candidates
    'solver.max_model_cand': 1,
    # Length (in bits) of bit vector
    'solver.bit_vec_len': 32,

    # Maximal number of operation vertices in a graph
    'graph.max_opr_num': 32,
    # Penalty coefficient on number of uses of a value
    'graph.use_penal': 4,
    # Number of records in diversity history of each operator
    'graph.num_div_record': 16,
    # Scale of the normalized diversity score
    'graph.div_score_scale': 0.2,
    # Probability of rejecting a non-unique operation
    'graph.reject_prob': 0.9,
    # Number of trials for generating one operation
    # For variadic operators, this is the maximal number of trials of adding a new input value
    'graph.opr_trials': 3,

    # Maximal kernel size of convolution
    'op.max_kernel': 3,
    # Maximal stride of convolution
    'op.max_stride': 2,
    # Maximal padding
    'op.max_padding': 2,
    # Maximal dilation rate of convolution
    'op.max_dilation': 2,
}

# Operators that are commonly supported by all the baselines
common_ops = [
    'sigmoid',
    'add',
    'subtract',
    'multiply',
    'maximum',
    'minimum',
    'reshape',
    'transpose',
    'concatenate',
    'strided_slice',
    'nn.relu',
    'nn.leaky_relu',
    'nn.prelu',
    'nn.softmax',
    'nn.conv1d',
    'nn.conv2d',
    'nn.max_pool2d',
    'nn.avg_pool2d',
    'nn.pad',
    'nn.batch_norm',
    'nn.dense',
    'nn.bias_add',
]

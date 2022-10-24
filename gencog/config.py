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
    'solver.max_model_cand': 4,
    # Length (in bits) of bit vector
    'solver.bit_vec_len': 32,

    # Maximal number of operation vertices in a graph
    'graph.max_opr_num': 32,
    # Penalty coefficient on number of uses of a value
    'graph.use_penal': 4,
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

# Operators that have correspondences with Keras layers in Muffin (ICSE'22)
common_ops = [
    'sigmoid',
    'tanh',
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
    'nn.bias_add',
    'nn.softmax',
    'nn.conv1d',
    'nn.conv2d',
    'nn.conv3d',
    'nn.conv2d_transpose',
    'nn.conv3d_transpose',
    'nn.max_pool1d',
    'nn.max_pool2d',
    'nn.max_pool3d',
    'nn.avg_pool1d',
    'nn.avg_pool2d',
    'nn.avg_pool3d',
    'nn.adaptive_max_pool1d',
    'nn.adaptive_max_pool2d',
    'nn.adaptive_max_pool3d',
    'nn.adaptive_avg_pool1d',
    'nn.adaptive_avg_pool2d',
    'nn.adaptive_avg_pool3d',
    'nn.upsampling',
    'nn.upsampling3d',
    'nn.pad',
    'nn.batch_norm',
    'nn.dense',
    'nn.batch_flatten',
]

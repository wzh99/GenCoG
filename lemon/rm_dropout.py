import datetime
from typing import *

import keras
import numpy as np
from keras.layers import Dropout

from lemon.mutation.mutation_utils import LayerUtils
from lemon.tools import utils


def _assert_indices(mutated_layer_indices: List[int], depth_layer: int):
    assert max(mutated_layer_indices) < depth_layer, "Max index should be less than layer depth"
    assert min(mutated_layer_indices) >= 0, "Min index should be greater than or equal to zero"


def _LC_and_LR_scan(model, mutated_layer_indices):
    layers = model.layers

    # the last layer should not be copied or removed
    mutated_layer_indices = np.arange(
        len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(mutated_layer_indices, len(layers))

    available_layer_indices = []
    for i, layer in enumerate(layers):
        if hasattr(layer, 'activation') and 'softmax' in layer.activation.__name__.lower():
            break
        if i in mutated_layer_indices:
            # InputLayer should not be copied or removed
            from keras.engine.input_layer import InputLayer
            if isinstance(layer, InputLayer):
                continue
            # layers with multiple input tensors can't be copied or removed
            if isinstance(layer.input, list) and len(layer.input) > 1:
                continue
            layer_input_shape = layer.input.shape.as_list()
            layer_output_shape = layer.output.shape.as_list()
            if layer_input_shape == layer_output_shape:
                available_layer_indices.append(i)
    np.random.shuffle(available_layer_indices)
    return available_layer_indices


def model_copy(model, mode=''):
    import keras
    suffix = '_copy_' + mode
    if model.__class__.__name__ == 'Sequential':
        new_layers = []
        for layer in model.layers:
            new_layer = LayerUtils.clone(layer)
            # new_layer.name += suffix
            new_layers.append(new_layer)
        new_model = keras.Sequential(layers=new_layers, name=model.name)
    else:
        new_model = utils.ModelUtils.functional_model_operation(model)

    s = datetime.datetime.now()
    new_model.set_weights(model.get_weights())
    e1 = datetime.datetime.now()
    td1 = e1 - s
    h, m, s = utils.ToolUtils.get_HH_mm_ss(td1)
    # print("Set model weights! {} hour,{} min,{} sec".format(h, m, s))
    del model
    return new_model


def RM_Dropout(model, mutated_layer_indices=None):
    LR_model = model_copy(model, 'LR')
    available_layer_indices = _LC_and_LR_scan(LR_model, mutated_layer_indices)

    if len(available_layer_indices) == 0:
        print('no appropriate layer to remove (input and output shape should be same)')
        return None

    # use logic: remove the first dropout layer
    remove_layer_index = None
    for idx in available_layer_indices:
        if isinstance(LR_model.layers[idx], Dropout):
            print(LR_model.layers[idx].name, type(LR_model.layers[idx].name))
            remove_layer_index = idx
            break
    if remove_layer_index is None:
        print('No dropout layer needs to delete')
        return None

    # remove_layer_index = available_layer_indices[-1]
    print('choose to remove layer {}'.format(LR_model.layers[remove_layer_index].name))
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(LR_model.layers):
            if i != remove_layer_index:
                new_layer = LayerUtils.clone(layer)
                # new_layer.name += '_copy'
                new_model.add(new_layer)
    else:
        new_model = utils.ModelUtils.functional_model_operation(LR_model, operation={
            LR_model.layers[remove_layer_index].name: lambda x, layer: x})

    # update weights
    assert len(new_model.layers) == len(model.layers) - 1
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LR'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in new_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


def check_dropout(model):
    for layer in model.layers:
        if isinstance(layer, Dropout) or 'dropout' in layer.name.lower():
            return False
    return True


if __name__ == '__main__':
    # 种子模型
    seed_model_names = ['alexnet-cifar10', 'densenet121-imagenet', 'inception.v3-imagenet',
                        'lenet5-fashion-mnist', 'lenet5-mnist', \
                        'resnet50-imagenet', 'vgg16-imagenet', 'vgg19-imagenet',
                        'xception-imagenet']  # 'mobilenet.1.00.224-imagenet' 去除dropout层时报错（应该是LEMON的copylayer函数有问题）
    # 处理前模型路径
    seed_model_path = './data/__origin_model/'
    # 处理后模型路径
    processed_model_path = './data/_origin_model/'

    for name in seed_model_names:
        model_path = '{}{}_origin.h5'.format(seed_model_path, name)

        origin_model = keras.models.load_model(model_path,
                                               custom_objects=utils.ModelUtils.custom_objects())

        # print('Model before processing:')
        # origin_model.summary()

        while True:
            processed_model = RM_Dropout(origin_model)
            if processed_model is None:
                processed_model = origin_model
                break
            else:
                origin_model = processed_model

        # print('Model After processing:')
        processed_model.summary()

        if check_dropout(processed_model):
            target_path = '{}{}_origin.h5'.format(processed_model_path, name)
            processed_model.save(target_path)
        else:
            print('########################################')
            print('Failed in checking processed {}.'.format(name))
            print('########################################')

import argparse
import datetime
import sys

import keras
import keras.backend as K
import numpy as np

from lemon.tools import utils


def clone(layer):
    custom_objects = utils.ModelUtils.custom_objects()
    layer_config = layer.get_config()
    # print(layer.name, layer_config)
    if 'data_format' in layer_config.keys():
        layer_config['data_format'] = 'channels_first'
    if 'batch_input_shape' in layer_config.keys() and len(layer_config['batch_input_shape']) == 4:
        new_batch_input_shape = (
            layer_config['batch_input_shape'][0], layer_config['batch_input_shape'][3],
            layer_config['batch_input_shape'][1], layer_config['batch_input_shape'][2])
        layer_config['batch_input_shape'] = new_batch_input_shape
    if 'axis' in layer_config.keys():
        layer_config['axis'] = 1
    if 'target_shape' in layer_config.keys() and len(layer_config['target_shape']) == 3:
        new_target_shape = (layer_config['target_shape'][2], layer_config['target_shape'][0],
                            layer_config['target_shape'][1])
        layer_config['target_shape'] = new_target_shape
    # print(layer.name, layer_config)
    if 'activation' in layer_config.keys():
        activation = layer_config['activation']
        if activation in custom_objects.keys():
            layer_config['activation'] = 'relu'
            clone_layer = layer.__class__.from_config(layer_config)
            clone_layer.activation = custom_objects[activation]
        else:
            clone_layer = layer.__class__.from_config(layer_config)
    else:
        clone_layer = layer.__class__.from_config(layer_config)
    return clone_layer


def functional_model_operation(model):
    input_layers = {}
    output_tensors = {}
    model_output = None
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in input_layers.keys():
                input_layers[layer_name] = [layer.name]
            else:
                if layer.name not in input_layers[layer_name]:
                    input_layers[layer_name].append(layer.name)

    assert len(model.input.shape) == 4
    assert len(model.inputs) == 1
    from keras.layers import Input

    # output_tensors[model.layers[0].name] = inputs[0]
    reshaped_input = K.reshape(model.input, (
        -1, model.input.shape[3], model.input.shape[1], model.input.shape[2]))
    inputs = [Input(
        shape=(model.input.shape[3], model.input.shape[1], model.input.shape[2]),
        tensor=reshaped_input,
        name=model.layers[0].name)]  # , name = model.input.name.split(':')[0]
    output_tensors[model.layers[0].name] = inputs[0]
    # print(output_tensors[model.layers[0].name])
    # inputs = [output_tensors[model.layers[0].name]]

    for layer in model.layers[1:]:
        layer_input_tensors = [output_tensors[l] for l in input_layers[layer.name]]
        if len(layer_input_tensors) == 1:
            layer_input_tensors = layer_input_tensors[0]

        # print(layer.name, input_layers[layer.name], layer_input_tensors)
        cloned_layer = clone(layer)
        x = cloned_layer(layer_input_tensors)

        output_tensors[layer.name] = x
        model_output = x

    return keras.Model(inputs=inputs, outputs=model_output)


def Change_channel(model):
    if model.__class__.__name__ == 'Sequential':
        new_layers = []
        for layer in model.layers:
            new_layer = clone(layer)
            new_layers.append(new_layer)
        new_model = keras.Sequential(layers=new_layers, name=model.name)
    else:
        new_model = functional_model_operation(model)

    s = datetime.datetime.now()
    # new_model.set_weights(model.get_weights())

    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        # layer_name = layer.name
        # if layer_name.endswith('_copy_LR'):
        #     key = layer_name[:-8]
        # else:
        #     key = layer_name
        # new_model_layers[key] = layer
        new_model_layers[layer.name] = layer

    tuples = []
    for layer_name in new_model_layers.keys():
        if layer_name.startswith('input'):
            continue
        # print('setting {}'.format(layer_name))
        # print(old_model_layers[layer_name].get_config())
        # print(new_model_layers[layer_name].get_config())
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            # print('shape_sw: {}, shape_w: {}'.format(shape_sw, shape_w))
            assert len(shape_sw) == len(shape_w)
            if len(shape_sw) == 3 and shape_sw[0] != shape_w[0]:
                w = np.reshape(w, shape_sw)
            else:
                assert shape_sw[0] == shape_w[0], '{}: {}/{}'.format(layer_name, shape_sw, shape_w)
            tuples.append((sw, w))
        # K.batch_set_value(tuples)
        # tuples = []

    K.batch_set_value(tuples)

    e1 = datetime.datetime.now()
    td1 = e1 - s
    h, m, s = utils.ToolUtils.get_HH_mm_ss(td1)
    # print("Set model weights! {} hour,{} min,{} sec".format(h, m, s))
    del model
    return new_model


if __name__ == '__main__':
    K.set_image_data_format("channels_first")

    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--model", type=str, help="model path")
    parse.add_argument("--save_path", type=str, help="model save path")
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    model_path = flags.model
    origin_model = keras.models.load_model(model_path,
                                           custom_objects=utils.ModelUtils.custom_objects())
    new_model = Change_channel(origin_model)

    if new_model is None:
        raise Exception("Error: Model channel change failed")
    else:
        new_model.save(flags.save_path)

    K.set_image_data_format("channels_last")

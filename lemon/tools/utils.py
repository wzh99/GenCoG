import datetime
import math
import os
import pickle
import warnings

import numpy as np

np.random.seed(20200501)
warnings.filterwarnings("ignore")
"""Set seed and Init cuda"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class ModelUtils:
    def __init__(self):
        pass

    @staticmethod
    def model_copy(model, mode=''):
        from ..mutation.mutation_utils import LayerUtils
        import keras
        suffix = '_copy_' + mode
        if model.__class__.__name__ == 'Sequential':
            new_layers = []
            for layer in model.layers:
                new_layer = LayerUtils.clone(layer)
                new_layer._name += suffix
                new_layers.append(new_layer)
            new_model = keras.Sequential(layers=new_layers, name=model.name + suffix)
        else:
            new_model = ModelUtils.functional_model_operation(model, suffix=suffix)

        s = datetime.datetime.now()
        new_model.set_weights(model.get_weights())
        e1 = datetime.datetime.now()
        td1 = e1 - s
        h, m, s = ToolUtils.get_HH_mm_ss(td1)
        # print("Set model weights! {} hour,{} min,{} sec".format(h, m, s))
        del model
        return new_model

    @staticmethod
    def functional_model_operation(model, operation=None, suffix=None):
        from ..mutation.mutation_utils import LayerUtils
        input_layers = {}
        output_tensors = {}
        model_output = None
        for layer in model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in input_layers.keys():
                    input_layers[layer_name] = [layer.name]
                else:
                    if layer.name not in input_layers[layer_name]:  # This condition is added by nie
                        input_layers[layer_name].append(layer.name)

        output_tensors[model.layers[0].name] = model.input

        for layer in model.layers[1:]:
            layer_input_tensors = [output_tensors[l] for l in input_layers[layer.name]]
            if len(layer_input_tensors) == 1:
                layer_input_tensors = layer_input_tensors[0]

            if operation is not None and layer.name in operation.keys():
                x = layer_input_tensors
                cloned_layer = LayerUtils.clone(layer)
                if suffix is not None:
                    cloned_layer._name += suffix
                x = operation[layer.name](x, cloned_layer)
            else:
                cloned_layer = LayerUtils.clone(layer)
                if suffix is not None:
                    cloned_layer._name += suffix
                x = cloned_layer(layer_input_tensors)

            output_tensors[layer.name] = x
            model_output = x

        import keras
        return keras.Model(inputs=model.inputs, outputs=model_output)

    @staticmethod
    def save_initial_weights(model):
        weights = model.get_weights()
        np.save('initial_weights.npy', weights)

    @staticmethod
    def load_initial_weights(model):
        weights = np.load('initial_weights.npy')
        model.set_weights(weights)
        return model

    @staticmethod
    def save_layers_output(path, layers_output):

        dirname = os.path.dirname(path)
        if len(dirname) > 0 and (not os.path.exists(dirname)):
            os.makedirs(dirname)
        with open(path, 'wb') as f:
            pickle.dump(layers_output, f)

    @staticmethod
    def load_layers_output(path):
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            layers_output = pickle.load(f)
        return layers_output

    @staticmethod
    def layers_output(model, input):
        from keras import backend as K
        # print(K.backend()+" in loadmodel")
        from keras.engine.input_layer import InputLayer
        get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [l.output for l in
                                       (model.layers[1:]
                                        if isinstance(model.layers[0], InputLayer)
                                        else model.layers)])
        if isinstance(model.layers[0], InputLayer):
            layers_output = [input]
            layers_output.extend(get_layer_output([input, 0]))
        else:
            layers_output = get_layer_output([input, 0])
        return layers_output

    @staticmethod
    def layers_input(model, input):
        inputs = [[input]]
        from keras import backend as K
        from keras.engine.input_layer import InputLayer
        for i, layer in enumerate(model.layers):
            if i == 0:
                continue
            if i == 1 and isinstance(model.layers[0], InputLayer):
                continue
            get_layer_input = K.function([model.layers[0].input, K.learning_phase()],
                                         layer.input if isinstance(layer.input, list) else [
                                             layer.input])
            inputs.append(get_layer_input([input, 0]))
        return inputs

    @staticmethod
    def generate_permutation(size_of_permutation, extract_portion):
        assert extract_portion <= 1
        num_of_extraction = math.floor(size_of_permutation * extract_portion)
        permutation = np.random.permutation(size_of_permutation)
        permutation = permutation[:num_of_extraction]
        return permutation

    @staticmethod
    def shuffle(a):
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        length = len(a)
        permutation = np.random.permutation(length)
        index_permutation = np.arange(length)
        shuffled_a[permutation] = a[index_permutation]
        return shuffled_a

    @staticmethod
    def compile_model(model, optimer, loss, metric: list):
        model.compile(optimizer=optimer,
                      loss=loss,
                      metrics=metric)
        return model

    @staticmethod
    def custom_objects():
        from ..mutation.mutation_utils import ActivationUtils
        objects = {}
        objects['no_activation'] = ActivationUtils.no_activation
        objects['leakyrelu'] = ActivationUtils.leakyrelu
        return objects

    @staticmethod
    def weighted_layer_indices(model):
        indices = []
        for i, layer in enumerate(model.layers):
            weight_count = layer.count_params()
            if weight_count > 0:
                indices.append(i)
        return indices

    @staticmethod
    def is_valid_model(inputs_backends, backends_nums, threshold=0.95):
        invalid_status_num = 0
        inputs_values = list(inputs_backends.values())
        # results like (1500,1) is valid
        if inputs_values[0].shape[1] == 1:
            return True
        else:
            for inputs in inputs_backends.values():
                indice_map = {}
                for input in inputs:
                    max_indice = np.argmax(input)
                    if max_indice not in indice_map.keys():
                        indice_map[max_indice] = 1
                    else:
                        indice_map[max_indice] += 1
                for indice in indice_map.keys():
                    if indice_map[indice] > len(inputs) * threshold:
                        invalid_status_num += 1

            return False if invalid_status_num == backends_nums else True


class ToolUtils:

    @staticmethod
    def get_HH_mm_ss(td):
        days, seconds = td.days, td.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return hours, minutes, secs


if __name__ == '__main__':
    pass

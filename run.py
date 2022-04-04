import random
from typing import Optional

import numpy as np

from muffin.model_generator import ModelGenerator
from muffin.selection import Roulette
from muffin.utils import layer_types, layer_conditions
from tvm_frontend import from_keras


class TrainingDebugger(object):

    def __init__(self, config: dict, use_heuristic: bool, generate_mode: str):
        super().__init__()
        self.__selector = Roulette(layer_types=layer_types,
                                   layer_conditions=layer_conditions,
                                   use_heuristic=use_heuristic)
        self.__model_generator = ModelGenerator(config['model'], self.__selector, generate_mode,
                                                config['model']['var'][
                                                    'weight_value_range'])

    def run_generation(self, model_info: Optional[dict] = None):
        '''随机生成模型和数据, 可指定现成模型信息
        '''
        # 随机生成model
        model = self.__model_generator.generate('', model_info=model_info)
        return model


config = {
    'model': {
        'var': {
            'tensor_dimension_range': (2, 5),
            'tensor_element_size_range': (1, 4),
            'weight_value_range': (-10.0, 10.0),
            'small_value_range': (0, 1),
            'vocabulary_size': 1001,
        },
        'node_num_range': (16, 16),
        'dag_io_num_range': (1, 3),
        'dag_max_branch_num': 2,
        'cell_num': 3,
        'node_num_per_normal_cell': 10,
        'node_num_per_reduction_cell': 2,
    },
    'backends': ['tensorflow'],
    'distance_threshold': 0,
}


def sample_model():
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import Conv2DTranspose

    x = Input(shape=(1, 2, 4))
    y = Conv2DTranspose(4, (1, 1), strides=(2, 2), data_format='channels_first')(x)
    model = Model(inputs=[x], outputs=[y])
    return model


def main():
    from tvm.relay.transform import AnnotateSpans

    case_num = 10
    use_heuristic = False  # 是否开启启发式规则
    generate_mode = 'template'  # seq\merge\dag\template

    debugger = TrainingDebugger(config, use_heuristic, generate_mode)

    for _ in range(case_num):
        try:
            model = debugger.run_generation()
            # model = sample_model()
        except ValueError as err:
            print(err.__cause__)
        input_shapes = {inp.name: (1,) + tuple(inp.shape.as_list()[1:]) for inp in model.inputs}
        mod, params = from_keras(model, shape=input_shapes)
        print(mod)
        mod = AnnotateSpans()(mod)
        print(mod)


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    main()

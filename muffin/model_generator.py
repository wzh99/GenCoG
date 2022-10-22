from functools import partial, reduce
from typing import Optional, Tuple, List

import keras
from keras.backend import set_image_data_format, int_shape

from keras_gen import KerasGenerator
from .dag import DAG
from .layer_info_generator import LayerInfoGenerator
from .model_template import ModelTemplate
from .selection import Roulette
from .utils import construct_layer_name, normal_layer_types, reduction_layer_types, get_layer_func, \
    layer_types, layer_conditions
from .variable_generator import VariableGenerator

set_image_data_format('channels_first')

config = {
    'var': {
        'tensor_dimension_range': (2, 5),
        'tensor_element_size_range': (1, 4),
        'weight_value_range': (-10.0, 10.0),
        'small_value_range': (0, 1),
        'vocabulary_size': 1001,
    },
    'node_num_range': (32, 32),
    'dag_io_num_range': (1, 3),
    'dag_max_branch_num': 2,
    'cell_num': 3,
    'node_num_per_normal_cell': 10,
    'node_num_per_reduction_cell': 2,
}


class MuffinGenerator(KerasGenerator):
    '''模型信息生成器
    '''

    def __init__(self, gen_mode: str):
        self._gen_mode = gen_mode
        self.__node_num_range = config['node_num_range']
        self.__dag_io_num_range = config['dag_io_num_range']
        self.__max_branch_num = config['dag_max_branch_num']
        self.__cell_num = config['cell_num']
        self.__node_num_per_normal_cell = config['node_num_per_normal_cell']
        self.__node_num_per_reduction_cell = config['node_num_per_reduction_cell']
        self.__random = VariableGenerator(config['var'])
        selector = Roulette(layer_types=layer_types,
                            layer_conditions=layer_conditions,
                            use_heuristic=False)
        self.__layer_generator = LayerInfoGenerator(self.__random, selector)
        self.__weight_range = config['var']['weight_value_range']

    def generate(self):
        '''生成模型信息, 可通过model_info指定模型信息，若不指定则随机生成

        返回值：
            json_path, input_shapes, model_id, exp_dir
        '''
        generate_mode = self._gen_mode
        # 若未规定现成结构
        node_num = 16 if generate_mode in ['dag', 'template'] else 32

        if generate_mode == 'seq':
            model_info, input_shapes, output_shapes, node_num = self.generate_seq_model(
                node_num=node_num)
        elif generate_mode == 'merge':
            model_info, input_shapes, output_shapes, node_num = self.generate_merge_model(
                node_num=node_num)
        elif generate_mode == 'dag':
            model_info, input_shapes, output_shapes, node_num = self.generate_dag_model(
                node_num=node_num)
        elif generate_mode == 'template':
            model_info, input_shapes, output_shapes, node_num = self.generate_template_model(
                cell_num=self.__cell_num,
                node_num_per_normal_cell=self.__node_num_per_normal_cell,
                node_num_per_reduction_cell=self.__node_num_per_reduction_cell)
        else:
            raise ValueError(f"UnKnown generate mode '{generate_mode}'")

        # Generate model from model info
        model = self.__generate_model(model_info)
        return model

    def __generate_layer(self, layer_info: dict):
        # 解析层信息
        layer_type, layer_args, pre_layers, ouput_shape = tuple(
            map(layer_info.get, ['type', 'args', 'pre_layers', 'output_shape']))
        layer = get_layer_func(layer_type)

        # 解析参数
        for k, v in layer_args.items():
            if k == 'layer':
                layer_args[k], _, _ = self.__generate_layer(v)  # 嵌套层需要递归生成
            elif k[-12:] == '_initializer':
                a, b = self.__weight_range
                if k == 'moving_variance_initializer':
                    layer_args[k] = keras.initializers.random_uniform(0, b)
                else:
                    layer_args[k] = keras.initializers.random_uniform(a, b)  # initializer

        return layer(**layer_args), pre_layers, ouput_shape

    def __generate_model(self, model_info: dict):
        # 加载模型结构数据
        input_id_list, output_id_list = model_info['input_id_list'], model_info['output_id_list']

        input_list, output_list, layer_dict = [], [], {}
        for layer_id, layer_info in model_info['model_structure'].items():  # 按拓扑排序遍历
            layer_id = int(layer_id)
            # 生成层
            layer, inbound_layers_idx, ouput_shape = self.__generate_layer(layer_info)

            # 层拼接
            if layer_id in input_id_list:
                layer_dict[layer_id] = layer  # input_object
                input_list.append(layer_dict[layer_id])

            else:
                inbound_layers = [layer_dict[i] for i in inbound_layers_idx]
                layer_dict[layer_id] = layer(
                    inbound_layers[0] if len(
                        inbound_layers) == 1 else inbound_layers)  # 对layers进行连接

            if layer_id in output_id_list:
                output_list.append(layer_dict[layer_id])

            # 检查形状
            if int_shape(layer_dict[layer_id]) != tuple(ouput_shape):
                raise Exception(
                    f"[Debug] layer_id: {layer_id} expected shape: {tuple(ouput_shape)}  "
                    f"actual shape: {int_shape(layer_dict[layer_id])}")

        return keras.Model(inputs=input_list, outputs=output_list)

    def generate_seq_model(self, node_num: int, start_id: int = 0,
                           pre_layer_id: Optional[int] = None, pre_layer_type: Optional[str] = None,
                           input_shape: Optional[Tuple[Optional[int]]] = None,
                           output_shape: Optional[Tuple[Optional[int]]] = None,
                           pool: Optional[list] = None, cell_type: str = ''):
        '''随机生成简单的链式型结构, 返回描述模型结构的dict以及模型的input_shape_list、output_shape_list

        参数:
            node_num: 结点个数, 若指定output_shape，则要求node_num >= 3（不包括input_object)
            start_id: 结点index的起始id, 默认从0开始
            input_shape：为生成的链式模型指定输入形状，若为None则表示不指定
            output_shape：为生成的链式模型指定输出形状，若为None则表示不指定
        返回值：
            (model_info_dict, input_shape, output_shape)
        '''
        if pre_layer_id is not None and input_shape is None:
            raise ValueError("input_shape of seq model should be provided.")

        model_structure = {}
        cur_shape = input_shape
        pre_layers = [] if pre_layer_id is None else [pre_layer_id]
        layer_type = pre_layer_type
        layer_name = None
        skip = 0
        i = start_id
        while i < start_id + node_num:
            if not pre_layers:  # 输入层
                layer_type, layer_args, cur_shape = self.__random.input_object(shape=cur_shape)
                input_shape = cur_shape  # 记录输入形状

            # 若指定output_shape则最后三层固定为flatten、dense和reshape
            elif output_shape and i >= start_id + node_num - 3:
                if i == start_id + node_num - 3 and len(cur_shape) <= 2:  # 不需要接Flatten层
                    skip += 1
                    i += 1
                    continue
                last_three_layers = [self.__layer_generator.layer_infos.flatten_layer,
                                     partial(self.__layer_generator.layer_infos.dense_layer,
                                             units=reduce(lambda x, y: x * y, output_shape[1:])),
                                     partial(self.__layer_generator.layer_infos.reshape_layer,
                                             output_shape=output_shape)]
                layer_type, layer_args, cur_shape = last_three_layers[
                    i - (start_id + node_num - 3)](cur_shape)

            else:  # 中间层
                layer_type, layer_args, cur_shape = self.__layer_generator.generate(
                    cur_shape,
                    last_layer=layer_type,
                    pool=pool
                )
                if layer_type is None:
                    skip += 1
                    i += 1
                    continue
            j = i - skip
            layer_name = construct_layer_name(j, layer_type, cell_type)
            # print(f"{layer_name}: {cur_shape}")

            # 形状太大的就抛出错误
            if self.__shape_too_big(cur_shape):
                raise ValueError(f"Invalid shape: {cur_shape}")

            model_structure[j] = dict(type=layer_type,
                                      args=dict(**layer_args, name=layer_name),
                                      pre_layers=pre_layers,
                                      output_shape=cur_shape)
            pre_layers = [j]
            i += 1

        return (
            dict(model_structure=model_structure,
                 input_id_list=[start_id],
                 output_id_list=[start_id + node_num - skip - 1]),
            {construct_layer_name(start_id, 'input_object',
                                  cell_type): input_shape} if pre_layer_id is None else {},
            {layer_name: cur_shape},
            node_num - skip
        )

    def generate_merge_model(self, node_num: int, start_id: int = 0,
                             pre_layer_ids: Optional[List[list]] = None,
                             pre_layer_types: Optional[List[str]] = None,
                             input_shapes: Optional[list] = None,
                             fin_output_shape: Optional[tuple] = None,
                             pool: Optional[list] = None, cell_type: str = ''):
        '''随机生成merge型结构, 返回描述模型结构的dict以及模型的input_shape_list、output_shape_list

        参数:
            node_num: 结点个数, 要求node_num >= 6(不包括input_object)
        返回值：
            (model_info_dict, input_shape_list, output_shape)
        '''
        node_num_list = self.__divide_node_num(node_num=node_num - 1,  # 参数这里是除去merge层
                                               inputs_num=(None if input_shapes is None else len(
                                                   input_shapes)),
                                               full_model=(pre_layer_ids is None))

        # 特殊处理
        merge_output_shape = fin_output_shape if fin_output_shape is not None and node_num_list[
            -1] == 0 else (
            self.__random.shape(
                dim=self.__random.randint_in_range((3, 5))) if cell_type == 'reduction' else None
        )

        layer_type, layer_args, merge_input_shape_list, merge_output_shape = self.__layer_generator.generate_merging_layer(
            input_num=len(node_num_list) - 1,
            output_shape=merge_output_shape)  # 参数这里是考虑merge层后无链式模型的情形

        model_structure = {}
        input_id_list = []
        input_shape_dict = {}
        merge_pre_layers = []
        cur_id = start_id

        # merge前的链式层
        for idx, (seq_node_num, seq_output_shape) in enumerate(
                zip(node_num_list[:-1], merge_input_shape_list)):
            model_info, input_shape, _, seq_node_num = self.generate_seq_model(
                node_num=seq_node_num,
                start_id=cur_id,
                pre_layer_id=(None if pre_layer_ids is None else pre_layer_ids[idx]),
                pre_layer_type=(None if pre_layer_ids is None else pre_layer_types[idx]),
                input_shape=(None if input_shapes is None else input_shapes[idx]),  # 指定的input_shape
                output_shape=seq_output_shape,
                pool=pool,
                cell_type=cell_type)
            model_structure.update(model_info['model_structure'])
            input_id_list += model_info['input_id_list']
            input_shape_dict.update(input_shape)
            merge_pre_layers.append(cur_id + seq_node_num - 1)
            cur_id += seq_node_num

        # merge层
        layer_name = construct_layer_name(cur_id, layer_type, cell_type)
        # print(f"{layer_name}: {merge_output_shape}")

        if self.__shape_too_big(merge_output_shape):
            raise ValueError("Shape too big!!")

        model_structure[cur_id] = dict(type=layer_type,
                                       args=dict(**layer_args, name=layer_name),
                                       pre_layers=merge_pre_layers,
                                       output_shape=merge_output_shape)
        cur_id += 1

        # merge后的链式层
        cur_shape = merge_output_shape
        output_shapes_dict = {layer_name: cur_shape}

        model_info, _, output_shapes_dict, seq_node_num = self.generate_seq_model(
            node_num=node_num_list[-1],
            start_id=cur_id,
            pre_layer_id=cur_id - 1,
            pre_layer_type=layer_type,
            input_shape=cur_shape,
            output_shape=fin_output_shape,
            pool=pool,
            cell_type=cell_type)
        model_structure.update(model_info['model_structure'])
        cur_id += seq_node_num

        return (
            dict(model_structure=model_structure,
                 input_id_list=input_id_list,
                 output_id_list=[cur_id - 1]),
            input_shape_dict,
            output_shapes_dict,
            cur_id - start_id
        )

    def generate_dag_model(self, node_num: int, start_id: int = 0,
                           input_shapes: Optional[list] = None,
                           output_shapes: Optional[list] = None,
                           pre_layer_ids: Optional[list] = None,
                           pre_layer_types: Optional[list] = None,
                           pool: Optional[list] = None, cell_type: str = ''):
        if input_shapes is None:
            input_shapes = [self.__random.shape() for _ in
                            range(self.__random.randint_in_range(self.__dag_io_num_range))]
        if output_shapes is None:
            output_shapes = [self.__random.shape() for _ in
                             range(self.__random.randint_in_range(self.__dag_io_num_range))]

        dag = DAG(node_num, input_shapes, output_shapes, max_branch_num=self.__max_branch_num)
        model_structure = {}
        input_id_list, output_id_list = [], []
        input_shapes_dict = {}
        output_shapes_dict = {}
        id_offset = 0
        total_num = 0
        input_idx = 0

        for node in dag.nodes:
            cur_id = start_id + node.id + id_offset
            if node.is_merging:
                plus_num = 3 * len(node.inbound_nodes)
                sub_model_info, _, sub_output_shapes_dict, sub_node_num = self.generate_merge_model(
                    node_num=plus_num + 1,
                    start_id=cur_id,
                    pre_layer_ids=[n1.id for n1 in node.inbound_nodes],
                    pre_layer_types=[n1.type for n1 in node.inbound_nodes],
                    input_shapes=[n1.output_shape for n1 in node.inbound_nodes],
                    fin_output_shape=node.output_shape,
                    pool=pool,
                    cell_type=cell_type)
            else:
                if node.is_input:
                    plus_num = 0
                    pre_layer_id = pre_layer_ids[input_idx] if pre_layer_ids is not None else None
                    pre_layer_type = pre_layer_types[
                        input_idx] if pre_layer_types is not None else None
                    input_idx += 1
                else:
                    plus_num = 0 if node.output_shape is None else 3
                    pre_layer_id = node.inbound_nodes[0].id
                    pre_layer_type = node.inbound_nodes[0].type

                sub_model_info, _, sub_output_shapes_dict, sub_node_num = self.generate_seq_model(
                    node_num=plus_num + 1,
                    start_id=cur_id,
                    pre_layer_id=pre_layer_id,
                    pre_layer_type=pre_layer_type,
                    input_shape=node.output_shape if node.is_input else node.inbound_nodes[
                        0].output_shape,
                    output_shape=None if pre_layer_ids is not None and node.is_input else node.output_shape,
                    pool=pool,
                    cell_type=cell_type)
            if sub_node_num:
                model_structure.update(sub_model_info['model_structure'])

                # 更新node信息
                node.id = cur_id + sub_node_num - 1
                node.output_shape = list(sub_output_shapes_dict.values())[0]
                node.type = list(sub_model_info['model_structure'].values())[-1]['type']
                layer_name = list(sub_model_info['model_structure'].values())[-1]['args']['name']
                id_offset += sub_node_num - 1
                total_num += sub_node_num

                if node.is_input:
                    input_id_list.append(node.id)
                    input_shapes_dict[layer_name] = node.output_shape

                if node.is_output:
                    output_id_list.append(node.id)
                    output_shapes_dict[layer_name] = node.output_shape
            else:  # 这个node被跳过的情形 (只存在于seq model)
                id_offset += sub_node_num - 1
                node.remove()

        return (
            dict(model_structure=model_structure,
                 input_id_list=input_id_list,
                 output_id_list=output_id_list),
            input_shapes_dict,
            output_shapes_dict,
            total_num
        )

    def generate_template_model(self, cell_num: int, node_num_per_normal_cell: int,
                                node_num_per_reduction_cell: int,
                                template_type: Optional[str] = None,
                                input_shape: Optional[Tuple[Optional[int]]] = None,
                                output_shape: Optional[Tuple[Optional[int]]] = None):
        if template_type is None:
            template_type = self.__random.choice(['T1'])

        model_template = ModelTemplate(self.__random, template_type=template_type,
                                       cell_num=cell_num,
                                       node_num_per_normal_cell=node_num_per_normal_cell,
                                       node_num_per_reduction_cell=node_num_per_reduction_cell,
                                       input_shape=input_shape, output_shape=output_shape)
        model_structure = {}
        input_id_list, output_id_list = [], []
        input_shapes_dict = {}
        output_shapes_dict = {}
        cur_id = 0
        pre_layer_ids = None
        pre_layer_types = None

        for cell in model_template.cells:
            pool = normal_layer_types if cell.type == 'normal' else reduction_layer_types

            sub_model_info, sub_input_shapes_dict, sub_output_shapes_dict, sub_node_num = self.generate_dag_model(
                node_num=cell.node_num,
                start_id=cur_id,
                input_shapes=cell.input_shapes,
                output_shapes=cell.output_shapes,
                pre_layer_ids=pre_layer_ids,
                pre_layer_types=pre_layer_types,
                pool=pool,
                cell_type=cell.type)

            model_structure.update(sub_model_info['model_structure'])

            # 更新信息
            pre_layer_ids = sub_model_info['output_id_list']
            pre_layer_types = [sub_model_info['model_structure'][idx]['type'] for idx in
                               pre_layer_ids]
            cur_id += sub_node_num

            if cell.is_input:
                input_id_list += sub_model_info['input_id_list']
                input_shapes_dict.update(sub_input_shapes_dict)

            if cell.is_output:
                output_id_list += sub_model_info['output_id_list']
                output_shapes_dict.update(sub_output_shapes_dict)

        cur_shape = list(output_shapes_dict.values())[-1]
        if len(cur_shape) > 2:
            flatten_layer = self.__layer_generator.layer_infos.flatten_layer
            layer_name, cur_id, cur_shape = self.add_single_layer(model_structure, flatten_layer,
                                                                  cur_id, cur_shape)

        # dense
        dense_layer = partial(self.__layer_generator.layer_infos.dense_layer, units=(
            model_template.output_shape[-1] if len(model_template.output_shape) > 1 else 1))
        layer_name, cur_id, cur_shape = self.add_single_layer(model_structure, dense_layer, cur_id,
                                                              cur_shape)

        # softmax
        softmax_layer = self.__layer_generator.layer_infos.softmax_layer
        layer_name, cur_id, cur_shape = self.add_single_layer(model_structure, softmax_layer,
                                                              cur_id, cur_shape)

        return (
            dict(model_structure=model_structure,
                 input_id_list=input_id_list,
                 output_id_list=[cur_id - 1]),
            input_shapes_dict,
            {layer_name: cur_shape},
            cur_id
        )

    def add_single_layer(self, model_structure, layer, layer_id, input_shape):
        layer_type, layer_args, cur_shape = layer(input_shape)
        layer_name = construct_layer_name(layer_id, layer_type, '')
        pre_layers = [layer_id - 1]
        layer_info = dict(type=layer_type,
                          args=dict(**layer_args, name=layer_name),
                          pre_layers=pre_layers,
                          output_shape=cur_shape)
        # print(f"{layer_name}: {cur_shape}")
        model_structure[layer_id] = layer_info
        return layer_name, layer_id + 1, cur_shape

    def __divide_node_num(self, node_num: int, inputs_num: Optional[int] = None,
                          full_model: bool = True):
        '''要求分割后除了最后一个元素外 都>=3(不含input_objects)
        '''
        min_len = 4 if full_model else 3

        if inputs_num is None:
            if node_num < min_len * 2:
                raise Exception("node num is not sufficient.")

            res_list = [self.__random.randint_in_range([min_len, node_num // 2]) for _ in range(2)]
            remain = node_num - sum(res_list)

            while remain >= min_len and self.__random.boolean():  # 一定概率不继续分割
                new_num = self.__random.randint_in_range([min_len, remain])
                res_list.append(new_num)
                remain -= new_num
        else:
            if node_num < inputs_num * min_len:
                raise Exception(
                    f"node num is not sufficient. node_num: {node_num} inputs_num: {inputs_num} min_len: {min_len}")

            res_list = [self.__random.randint_in_range([min_len, node_num // inputs_num]) for _ in
                        range(inputs_num)]
            remain = node_num - sum(res_list)

        res_list.append(remain)  # 后续链长可以为0
        return res_list

    def __shape_too_big(self, cur_shape):
        temp = 1
        for e in cur_shape[1:]:
            if e > 1e6 or e <= 0:
                return True
            temp *= e
        return temp > 1e8

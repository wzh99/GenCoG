import random
import string
import types
from typing import List, Union

import tvm


def filter_target(target):
    return lambda f: f.with_attr("target", target)


__TIR_PASS_NAME__ = [
    'Apply',
    'BF16CastElimination',
    'BF16Legalize',
    'BF16Promote',
    'BF16TypeLowering',
    'CoProcSync',
    'CombineContextCall',
    'CompactBufferAllocation',
    'ConvertBlocksToOpaque',
    # 'DecorateDeviceScope',
    # 'Filter',
    'FlattenBuffer',
    'HoistIfThenElse',
    'InferFragment',
    # 'InjectCopyIntrin',
    'InjectDoubleBuffer',
    'InjectPrefetch',
    'InjectVirtualThread',
    'InstrumentBoundCheckers',
    'LegalizePackedCalls',
    'LiftAttrScope',
    'LoopPartition',
    'LowerCustomDatatypes',
    'LowerDeviceStorageAccessInfo',
    'LowerInitBlock',
    'LowerIntrin',
    'LowerMatchBuffer',
    'LowerTVMBuiltin',
    'LowerThreadAllreduce',
    'LowerWarpMemory',
    # 'MakePackedAPI',
    # 'MakeUnpackedAPI',
    'MergeDynamicSharedMemoryAllocations',
    'NarrowDataType',
    'PlanAndUpdateBufferAllocationLocation',
    'RemoveNoOp',
    'RewriteUnsafeSelect',
    'Simplify',
    'SkipAssert',
    'SplitHostDevice',
    'StorageFlatten',
    'StorageRewrite',
    'TextureFlatten',
    'ThreadSync',
    'UnifyThreadBinding',
    'UnrollLoop',
    'VectorizeLoop',
    'VerifyMemory'
]


class PassNode:
    def __init__(self, name, tvm_pass) -> None:
        self.name = name
        self.tvm_pass = tvm_pass
        self.need_arguments = False
        self.disable = False
        self.dependence = None

    def mutate_on_the_fly(self):
        if not self.need_arguments:
            return self.tvm_pass()
        else:
            if isinstance(self.args, list) or isinstance(self.args, tuple):
                return self.tvm_pass(random.choice(self.args))
            elif self.args == int:
                return self.tvm_pass(random.randint(8, 128))
            elif self.args == str:
                letters = string.ascii_letters + string.digits
                return self.tvm_pass(
                    ''.join([random.choice(letters) for _ in range(random.randint(1, 128))]))
            else:
                return self.tvm_pass(self.args)

    def mutate_and_get_pass_info(self) -> List[Union[tvm.transform.Pass, dict]]:
        if not self.need_arguments:
            return [self.tvm_pass(), {"pass_name": self.name, "arg_type": "None", "args": None}]
        else:
            if isinstance(self.args, list) or isinstance(self.args, tuple):
                args = random.choice(self.args)
                # arg_type = None
                if args is None:
                    arg_type = "None"
                elif type(args) == int:
                    arg_type = "int"
                elif type(args) == str:
                    arg_type = "str"
                else:
                    assert type(args) == bool
                    arg_type = "bool"
            elif isinstance(self.args, types.FunctionType):
                arg_type = "function"
                args = "filter_target"
                return [self.tvm_pass(self.args),
                        {"pass_name": self.name, "arg_type": arg_type, "args": args}]
            elif self.args == str:
                arg_type = "str"
                letters = string.ascii_letters + string.digits
                args = ''.join([random.choice(letters) for _ in range(random.randint(1, 128))])
            else:
                assert self.args == int
                arg_type = "int"
                args = random.randint(8, 128)
            return [self.tvm_pass(args),
                    {"pass_name": self.name, "arg_type": arg_type, "args": args}]

    def mutate_use_record(self, arg_type, args=None):
        if not self.need_arguments:
            assert arg_type == "None"
            return self.tvm_pass()
        else:
            if arg_type == "None":
                assert args is None
                return self.tvm_pass(None)
            elif arg_type == "function":
                assert isinstance(self.args, types.FunctionType)
                return self.tvm_pass(self.args)  # filter_target
            else:
                assert arg_type in {"str", "bool", "int"}
                return self.tvm_pass(args)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'

    def __repr__(self) -> str:
        return str(self)


class PassNodeFactory:
    def __init__(self, pass_module, pass_names) -> None:
        self.pass_nodes = {}
        for pass_name in pass_names:
            tvm_pass = getattr(pass_module, pass_name)
            pass_node = PassNode(pass_name, tvm_pass)
            self.pass_nodes[pass_name] = pass_node

    def get_pass_nodes(self):
        return self.pass_nodes

    def get_pass_node(self, pass_name):
        if pass_name not in self.pass_nodes.keys():
            return None
        return self.pass_nodes[pass_name]

    def __getitem__(self, name):
        return self.get_pass_node(name)


class PassDependenceGraph:
    def __init__(self, target=tvm.target.Target('llvm')) -> None:
        tir_pass_factory = PassNodeFactory(tvm.tir.transform.transform, __TIR_PASS_NAME__)
        self.tir_pass_nodes = tir_pass_factory.get_pass_nodes()
        self.target = target
        self.init_graph()

    def init_graph(self):
        arguments_tir_passes = [
            ('Apply', filter_target(self.target)),
            # 'Filter',           # function
            ('HoistIfThenElse', ("basic", None)),
            ('LiftAttrScope', str),  # string
            ('NarrowDataType', (32, 64)),  # int
            ('StorageFlatten', (32, 64)),  # int
            ('ThreadSync', ('global', 'shared', 'warp', 'local')),
            # global shared warp local wmma.matrix_a wmma.matrix_b wmma.accumulator
            ('VectorizeLoop', (True, False))  # True, False
        ]

        dependence_nodes = [
            ('Apply', 'LowerCustomDatatypes'),
            # ('Apply', 'MakePackedAPI'),
            # ('Apply', 'MakeUnpackedAPI'),
            ('Apply', 'LowerIntrin'),
            ('Apply', 'LowerThreadAllreduce'),
            ('Apply', 'LowerWarpMemory'),
            ('Apply', 'SplitHostDevice'),
            ('Apply', 'VerifyMemory'),
        ]

        for arguments_tir_pass, args in arguments_tir_passes:
            if arguments_tir_pass in self.tir_pass_nodes.keys():
                self.tir_pass_nodes[arguments_tir_pass].need_arguments = True
                self.tir_pass_nodes[arguments_tir_pass].args = args

        for node1, node2 in dependence_nodes:
            node1 = self.tir_pass_nodes[node1]
            node2 = self.tir_pass_nodes[node2]
            node2.dependence = node1

        self.all_tir_pass_nodes = [node for node in self.tir_pass_nodes.values() if
                                   not node.disable]
        self.root_candidates = [node for node in self.all_tir_pass_nodes if node.dependence == None]

    def get_concrete_pass(self, pass_info) -> tvm.transform.Pass:
        pass_name, arg_type, args = pass_info['pass_name'], pass_info['arg_type'], pass_info['args']
        return self.tir_pass_nodes[pass_name].mutate_use_record(arg_type, args)

    def random_tir_passes(self, length=1):
        return random.sample(self.all_tir_pass_nodes, min(length, len(self.all_tir_pass_nodes)))

    def indices_to_passes(self, indices):
        ret = []
        for idx in indices:
            ret.append(self.all_tir_pass_nodes[idx])
        return ret

    def insert_dependency(self, pass_nodes):
        old_pass_nodes = pass_nodes.copy()
        idx = 0
        while idx < len(pass_nodes):
            node = pass_nodes[idx]
            if node.dependence != None:
                pass_nodes.insert(idx, node.dependence)
                idx += 1
            idx += 1
        return pass_nodes

    def recover(self, names):
        pass_nodes = [self.tir_pass_nodes[name] for name in names if name in self.tir_pass_nodes]
        # return [pass_node.mutate() for pass_node in pass_nodes]
        return pass_nodes

    def export_name(self, pass_nodes):
        return [pass_node.name for pass_node in pass_nodes]

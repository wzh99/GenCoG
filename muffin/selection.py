from typing import List

import numpy as np


class Roulette(object):
    class Element(object):
        def __init__(self, name: str, selected: int = 0):
            self.name = name
            self.selected = selected

        def record(self):
            self.selected += 1

        @property
        def score(self):
            return 1.0 / (self.selected + 1)

    def __init__(self, layer_types: List[str], layer_conditions: dict, use_heuristic: bool = True):
        self.__pool = {name: self.Element(name=name) for name in layer_types}
        self.__layer_conditions = layer_conditions
        self.__use_heuristic = use_heuristic

    def update(self, name):
        if name == 'input_object':
            return
        for n, el in self.__pool.items():
            if n == name:
                el.record()
                return

    def coverage(self):
        selected_map = {name: el.selected for name, el in self.__pool.items()}
        cnt = 0
        for selected in selected_map.values():
            if selected > 0:
                cnt += 1
        coverage_rate = cnt / len(selected_map)
        return coverage_rate, selected_map

    def choose_element(self, pool: List[str], **kwargs):
        candidates = []
        _sum = 0
        for el_name in pool:
            cond = self.__layer_conditions.get(el_name, None)
            if cond is None or cond(**kwargs):  # available的layer
                candidates.append(self.__pool[el_name])
                _sum += self.__pool[el_name].score

        if self.__use_heuristic:
            rand_num = np.random.rand() * _sum
            for el in candidates:
                if rand_num < el.score:
                    return el.name
                else:
                    rand_num -= el.score
        else:
            return np.random.choice(candidates).name

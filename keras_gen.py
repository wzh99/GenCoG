from abc import ABC

from keras import Model


class KerasGenerator(ABC):
    def generate(self) -> Model:
        raise NotImplemented()

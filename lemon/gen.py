import os
import random
import shutil
import subprocess

import keras
from keras import Model

from keras_gen import KerasGenerator
from lemon.tools.utils import ModelUtils


class LemonGenerator(KerasGenerator):
    def __init__(self) -> None:
        # Mutators
        self.mutate_ops = ['ARem', 'ARep', 'LA', 'LC', 'LR', 'LS',
                           'MLA']  # 'WS', 'GF', 'NEB', 'NAI', 'NS' : inner mutators
        # Seed models
        self.seed_model_names = ['alexnet-cifar10', 'densenet121-imagenet', 'inception.v3-imagenet',
                                 'lenet5-fashion-mnist', 'lenet5-mnist', 'resnet50-imagenet',
                                 'xception-imagenet']

        self.seed_model_path = 'data/_origin_model'

        # Clone models to the output directory
        self.mut_model_dir = 'out/lemon_mut'
        if not os.path.exists(self.mut_model_dir):
            os.mkdir(self.mut_model_dir)
        for model_file in os.listdir(self.seed_model_path):
            shutil.copy(os.path.join(self.seed_model_path, model_file), self.mut_model_dir)

        # Mutation epochs
        self.min_epochs = 1
        self.max_epochs = 5

    def generate(self) -> Model:
        # Number of mutations
        epochs = 1

        # Read seed model
        model_name = random.choice(self.seed_model_names)
        model_path = os.path.join(self.mut_model_dir, f'{model_name}_origin.h5')

        for epoch in range(epochs):
            # Choose mutator
            mutator = random.choice(self.mutate_ops)
            # print('epoch {}: {}'.format(epoch, mutator))
            # Apply mutation
            subprocess.run([
                'python3', '-m', 'lemon.mutation.model_mutation_generators',
                '--model', model_path, '--mutate_op', mutator, '--save_path', model_path
            ], stderr=open(os.devnull, 'w'))
            # mutate_status = os.system(
            #     "python3 -m lemon.mutation.model_mutation_generators --model {} --mutate_op {} "
            #     "--save_path {}".format(model_path, mutator, model_path))

        # Convert to channel-first
        cvt_res = subprocess.run([
            'python3', '-m', 'lemon.change_channel',
            '--model', model_path, '--save_path', 'out/tmp.h5'
        ], stderr=open(os.devnull, 'w'))
        # channel_status = os.system("python3 -m lemon.change_channel --model {} --save_path
        # out/tmp.h5".format(model_path))
        if cvt_res.returncode == 0:
            generated_model = keras.models.load_model('out/tmp.h5', compile=False,
                                                      custom_objects=ModelUtils.custom_objects())
            os.remove('out/tmp.h5')
        else:
            raise RuntimeError('Cannot convert layout to channel-first')

        return generated_model

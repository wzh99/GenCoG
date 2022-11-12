import os
from argparse import ArgumentParser

import numpy as np
from numpy.random import Generator, PCG64

from gencog.debug import ModuleRunner, ModuleError

# Parse arguments
parser = ArgumentParser()
parser.add_argument('-d', '--directory', type=str, help='Case directory.')
parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed.')
parser.add_argument('-e', '--error', action='store_true', help='Report error.')
parser.add_argument('-r', '--result', action='store_true', help='Write results.')
args = parser.parse_args()

# Initialize runner
rng = Generator(PCG64(seed=args.seed))
runner = ModuleRunner(rng)

# Parse and run source code
with open(os.path.join(args.directory, 'code.txt'), 'r') as f:
    code = f.read()
try:
    all_results = runner.run(code)
except ModuleError as err:
    all_results = []
    if args.error:
        err.report(args.directory)
    exit(1)

# Save results
if args.result:
    for opt_level, level_results in enumerate(all_results):
        np.savez(os.path.join(args.directory, f'O{opt_level}.npz'), *level_results)

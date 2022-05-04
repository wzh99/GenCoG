import os
from argparse import ArgumentParser

from numpy.random import Generator, PCG64

from typefuzz.debug import ModuleRunner, ModuleError

# Parse arguments
parser = ArgumentParser()
parser.add_argument('-d', '--directory', type=str, help='Case directory.')
parser.add_argument('-s', '--seed', type=int, help='Random seed.')
args = parser.parse_args()

# Initialize runner
rng = Generator(PCG64(seed=args.seed))
runner = ModuleRunner(rng)

# Parse and run source code
with open(os.path.join(args.directory, 'code.txt'), 'r') as f:
    code = f.read()
try:
    runner.run(code)
except ModuleError as err:
    err.report(args.directory)
    exit(1)

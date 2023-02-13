# GenCoG

## Introduction

GenCoG is a DSL-based approach to generating computation graphs for TVM testing. It contains (1)
GenCoGL, a domain-specific language for specifying type constraints of operators, and (2) an
incremental generation approach with expressivity-directed strategy and concolic constraint solving.

## Contents

* [`gencog`](gencog): Python package of GenCoG implementation;
* [`bug`](bug): [Bug list](bug/bug_list.md), triggering Python scripts and bug reports;
* [`lemon`](lemon): Adapted implementation of [LEMON](https://github.com/Jacob-yen/LEMON) evaluated
  in the paper;
* [`muffin`](muffin): Adapted implementation of [Muffin](https://github.com/library-testing/Muffin)
  evaluated in the paper;
* [`graphfuzz`](graphfuzz): Our implementation
  of [Luo et al.](https://ieeexplore.ieee.org/document/9401995/) by reusing the code in GenCoG;
* [`tvm_frontend`](tvm_frontend): Modified TVM Keras frontend with better support for channel-first
  layout;
* Several Python scripts for running experiments.

## Dependency

GenCoG is written in Python. Run `pip install -r requirements.txt` to get all dependencies. Before
running testing and evaluation scripts, create a subdirectory `out` in root directory of this
project to store all their outputs.

## Bug Detection

### Running Test

```shell
python run_test.py
```

A working directory `out/run-%Y%m%d-%H%M%S` will be created. Each generated program will be run in a
separate process. If the process exits abnormally, the test case will be kept and the error message
will also be stored. Otherwise, the case will be deleted.

### Case Deduplication

```shell
python dedup_case.py -d ${WORK_DIR}
```

It deduplicates the cases with similar error messages, which indicate that they may share the same
root cause.

### Case Reduction

```shell
python reduce_case.py -d ${WORK_DIR}
```

It reduces each test case to a possibly simpler graph with fewer vertices.

## Evaluation

We have provided scripts to reproduce the evaluation results of GenCoG in the paper. Luo et al. can
be evaluated with two graph models: `ws` (Watts-Strogatz) and `rn` (Residual Network). Muffin can be
evaluated with two modes: `dag` (chain structure with skips) and `template` (cell-base structure).

### Validity

GenCoG:

```shell
python relay_valid.py -g gencog -n 1000
```

Luo et al.:

```shell
python relay_valid.py -g graphfuzz -n 1000 -m {ws|rn}
```

LEMON:

```shell
python keras_valid.py -g lemon -n 1000
```

Muffin:

```shell
python keras_valid.py -g muffin -n 1000 -m {dag|template}
```

NNSmith:

```shell
python nnsmith_valid.py -n 1000
```

### Expressivity

GenCoG:

```shell
python relay_div.py -g gencog -l 50000
```

Data are saved to `out/gencog-${opset}-%Y%m%d-%H%M%S.txt`.

Luo et al.:

```shell
python relay_div.py -g graphfuzz -l 20000 --model {ws|rn}
```

Data are saved to `out/graphfuzz-${model}-%Y%m%d-%H%M%S.txt`.

LEMON:

```shell
python keras_div.py -g lemon -l 20000
```

Data are saved to `out/lemon-%Y%m%d-%H%M%S.txt`.

Muffin:

```shell
python keras_div.py -g muffin -l 20000 --mode {dag|template}
```

Data are saved to `out/muffin-${mode}-%Y%m%d-%H%M%S.txt`.

NNSmith:

```shell
python nnsmith_div.py -l 20000
```

Data are saved to `out/nnsmith-%Y%m%d-%H%M%S.txt`.

## Extension

### Write Constraint Specifications for New Operators

Refer to files in [`gencog/op`](gencog/op) for how to write constraint specification for an operator
and register it in `OpRegistry`.

### Support New DL Compilers

Type constraints of operators in different DL compilers are possibly different. Some specifications
may need to be rewritten.

A new code generator is also required for generating high-level IR for the new DL compiler, from the
in-memory graph representation of GenCoG. [`gencog/graph/relay.py`](gencog/graph/relay.py) is the
code generator for Relay. You can refer to this file to implement your own generator.

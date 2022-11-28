# GenCoG

## Introduction

GenCoG is a diversity-oriented, type-constrained approach to generating computation graphs for TVM
testing. It contains (1) GenCoGL, a domain-specific language for specifying type constraints of
operators, and (2) an incremental graph generation algorithm which iteratively solves type
constraints and generates valid and diverse computation graphs.

The manuscript of this work is submitted to ACM Transactions on Software Engineering and
Methodology (TOSEM) for review.

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

GenCoG is written in Python. Run `pip install -r requirements.txt` to get all dependencies. To
evaluate code coverage, another build of TVM
with [Gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html) should be provided.

Before running testing and evaluation scripts, create a subdirectory `out` in root directory of this
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

We have provided scripts to reproduce the evaluation results of GenCoG in the paper. GenCoG can be
evaluated on two operator sets: `muffin` (39 operators commonly supported by both methods)
and `all` (all the 62 operators currently covered by GenCoG). Luo et al. can be evaluated with two
graph models: `ws` (Watts-Strogatz) and `rn` (Residual Network). Muffin can be evaluated with two
modes: `dag` (chain structure with skips) and `template` (cell-base structure).

### Validity

GenCoG:

```shell
python relay_valid.py -g gencog -n 10000 --opset {muffin|all}
```

Luo et al.:

```shell
python relay_valid.py -g graphfuzz -n 10000 -m {ws|rn}
```

LEMON:

```shell
python keras_valid.py -g lemon -n 10000
```

Muffin:

```shell
python keras_valid.py -g muffin -n 10000 -m {dag|template}
```

### Diversity

GenCoG:

```shell
python relay_div.py -g gencog -l 50000 --opset {muffin|all}
```

Diversity data are saved to `out/gencog-${opset}-%Y%m%d-%H%M%S.txt`.

Luo et al.:

```shell
python relay_div.py -g graphfuzz -l 50000 --model {ws|rn}
```

Diversity data are saved to `out/graphfuzz-${model}-%Y%m%d-%H%M%S.txt`.

LEMON:

```shell
python keras_div.py -g lemon -l 50000
```

Diversity data are saved to `out/lemon-%Y%m%d-%H%M%S.txt`.

Muffin:

```shell
python keras_div.py -g muffin -l 50000 --mode {dag|template}
```

Diversity data are saved to `out/muffin-${mode}-%Y%m%d-%H%M%S.txt`.

### Coverage

First build TVM with Gcov. The build files should be stored in `build` subdirectory in the root
directory of TVM source.

GenCoG:

```shell
python relay_cov.py -r ${TVM_GCOV_ROOT} -g gencog -l 50000 -s 1000 --opset {muffin|all}
```

`TVM_GCOV_ROOT` is the root directory of TVM source containing Gcov build. Coverage files are saved
to `cov-gencog-${opset}-%Y%m%d-%H%M%S` directory. `cov.json` is the final coverage
summary. `data.txt` is the line coverage data over vertex budget.

Luo et al.:

```shell
python relay_cov.py -r ${TVM_GCOV_ROOT} -g graphfuzz -l 50000 -s 1000 --model {ws|rn}
```

Coverage files are saved to `cov-graphfuzz-${model}-%Y%m%d-%H%M%S` directory.

LEMON:

```shell
python keras_cov.py -r ${TVM_GCOV_ROOT} -g lemon -l 50000 -s 1000
```

Coverage files are saved to `cov-lemon-%Y%m%d-%H%M%S` directory.

Muffin:

```shell
python keras_cov.py -r ${TVM_GCOV_ROOT} -g muffin -l 50000 -s 1000 --mode {dag|template}
```

Coverage files are saved to `cov-muffin-${opset}-%Y%m%d-%H%M%S` directory.

Tzer: Please
check [the instructions](https://github.com/MatthewXY01/tzer/blob/v0.1-reproduce/src/Instructions_for_gcov-test.md)
in [this repository](https://github.com/MatthewXY01/tzer) for details.

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

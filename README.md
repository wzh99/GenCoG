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
* [`bug`](bug): Triggering Python scripts and reports of bugs detected by our work;
* [`muffin`](muffin): Modified implementation of [Muffin](https://github.com/library-testing/Muffin)
  which is compared with GenCoG in the paper;
* [`tvm_frontend`](tvm_frontend): Modified TVM Keras frontend with better support for channel-first
  layout;
* Several Python scripts for running experiments.

## Dependency

GenCoG is written in Python. Run `pip install -r requirements.txt ` to get all dependencies. To
evaluate code coverage, another build of TVM
with [Gcov](https://gcc.gnu.org/onlinedocs/gcc/Gcov.html) should be provided.

Before running testing and evaluation scripts, create a subdirectory `out` in root directory of this
project to store all their outputs.

## Bug Detection

### Running Test

```
python run_test.py
```

A working directory `out/run-%Y%m%d-%H%M%S` will be created. Each generated program will be run in a
separate process. If the process exits abnormally, the test case will be kept and the error message
will also be stored. Otherwise, the case will be deleted.

### Case Deduplication

```
python dedup_case.py -d=${WORK_DIR}
```

It deduplicates the cases with similar error messages, which indicate that they may share the same
root cause.

### Case Reduction

```
python reduce_case.py -d=${WORK_DIR}
```

It reduces each test case to a possibly simpler graph with fewer vertices.

## Evaluation

We have provided scripts to reproduce the evaluation results of both GenCoG and Muffin in the paper.
GenCoG can be evaluated on two operator sets: `muffin` (39 operators commonly supported by both
methods) and `all` (all the 62 operators currently covered by GenCoG). Muffin can be evaluated with
two modes: `dag` (chain structure with skips) and `template` (cell-base structure).

### Validity

GenCoG:

```
python gencog_valid.py -n=10000 --opset={muffin|all}
```

Muffin:

```
python muffin_valid.py -n=10000 --mode={dag|template}
```

### Diversity

GenCoG:

```
python gencog_div.py -l=50000 --opset={muffin|all}
```

Diversity data over time is saved to `out/gencog-${opset}-%Y%m%d-%H%M%S.txt`.

Muffin:

```
python muffin_div.py -l=50000 --mode={dag|template}
```

Diversity data over time is saved to `out/muffin-${mode}-%Y%m%d-%H%M%S.txt`.

### Coverage

First build TVM with Gcov. The build files should be stored in `build` subdirectory in the root
directory of TVM source.

GenCoG:

```
python gencog_cov.py -r=${TVM_GCOV_ROOT} -l=50000 -s=1000 --opset={muffin|all}
```

`TVM_GCOV_ROOT` is the root directory of TVM source containing Gcov build. Coverage files are saved
to `cov-gencog-${opset}-%Y%m%d-%H%M%S` directory. `cov.json` is the final coverage
summary. `data.txt` is the line coverage data over time.

Muffin:

```
python muffin_cov.py -r=${TVM_GCOV_ROOT} -l=50000 -s=1000 --mode={dag|template}
```

The output files are the same as GenCoG.

## Reuse

### Write Constraint Specifications for New Operators

Refer to files in [`gencog/op`](gencog/op) for how to write constraint specification for an
operator.

### Support New DL Compilers

Type constraints of operators in different DL compilers are possibly different. Some specifications
may need to be rewritten.

A new code generator is also required for generating high-level IR for the new DL compiler, from the
in-memory graph representation of GenCoG. [`gencog/graph/relay.py`](gencog/graph/relay.py) is the
code generator for Relay. You can refer to this file to implement your own generator.

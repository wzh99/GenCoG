# GenCoG

[Paper](https://dl.acm.org/doi/10.1145/3597926.3598105)
| [Slides](https://www.aliyundrive.com/s/tUbmidkvLav)
| [Artifact](https://doi.org/10.5281/zenodo.7955514)

## Introduction

GenCoG is a DSL-based approach to generating computation graphs for TVM testing. It contains (1)
GenCoGL, a domain-specific language for specifying type constraints of operators, and (2) an
incremental generation approach with expressivity-directed strategy and concolic constraint solving.

## Contents

* Implementation of GenCoG;
* Adapted version or reimplementation of baselines: [LEMON](https://github.com/Jacob-yen/LEMON)
  , [Muffin](https://github.com/library-testing/Muffin)
  , [Luo et al.](https://ieeexplore.ieee.org/document/9401995/),
  and [NNSmith](https://github.com/ise-uiuc/nnsmith).
* [Bug list](bug/bug_list.md), triggering Python scripts and bug reports.

## Dependency

First, make sure TVM is installed. GenCoG works on v0.8 and v0.9, and it may also support later
versions. Then, run `pip install -r requirements/core.txt` to get all dependencies of GenCoG.

If you want to run the experiments in the paper, run `pip install -r requirements/exp.txt` to
install dependencies of the baselines.

## Bug Detection

Please first create a subdirectory `out` in the root directory of this project to store all the
outputs.

### Running Test

```shell
python3 run_test.py
```

A working directory `out/run-%Y%m%d-%H%M%S` will be created. Each generated program will be run in a
separate process. If the process exits abnormally, the test case will be kept and the error message
will also be stored. Otherwise, the case will be deleted.

### Case Deduplication

```shell
python3 dedup_case.py -d ${WORK_DIR}
```

It deduplicates the cases with similar error messages, which indicate that they may share the same
root cause.

### Case Reduction

```shell
python3 reduce_case.py -d ${WORK_DIR}
```

It reduces each test case to a possibly simpler graph with fewer vertices.

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

## Citation

```bibtex
@inproceedings{wang2023gencog,
    author = {Zihan Wang, Pengbo Nie, Xinyuan Miao, Yuting Chen, Chengcheng Wan, Lei Bu, Jianjun Zhao},
    title = {GenCoG: A DSL-Based Approach to Generating Computation Graphs for TVM Testing},
    year = {2023},
    publisher = {ACM},
    address = {New York, NY, USA},
    doi = {10.1145/3597926.3598105},
    booktitle = {Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and 
    Analysis},
    numpages = {13},
    series = {ISSTA ’23}
}

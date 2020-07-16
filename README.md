# Catenary

![http://dataphys.org/list/gaudis-hanging-chain-models/](http://dataphys.org/list/wp-content/uploads/2015/01/IMG_5393_640.jpg)

Catenary is a research codebase exploring an implementation of the ideas in
Henry Lin's [Bootstraps to Strings](https://arxiv.org/abs/2002.08387) paper.

The code is organized as follows:

- The `mathematica/original` directory contains the original Mathematica code
  used to generate the figures in the paper, and some explorations from
  @sritchie.
- The rest of the notebooks in `mathematica` contain a library the generates the
  loop equations that we need to solve to solve matrix models.
- The `catenary` directory contains the Python code we've written to solve the
  single matrix model. One challenge will be extending this to search over
  systems with lots of parameters

# Call for Collaboration

Here's the general idea of the method in the paper:

Given some model of the form

$$V(A) = {1 \over 2} A^2$$

## 1. Solving the Loop Equations

This problem uses the code in `mathematica/MatrixModels.mb`

## 2. Critical Point Search

Optimization loop.

### Gradient Ascent on Minimum Eigenvalue

### Finding Critical Points

# Tools

This section describes how to install the various software tools that we use to attack Catenary.

## Mathematica

## JAX / Python

# Trouble?

Get in touch with [samritchie@x.team](mailto:samritchie@x.team).

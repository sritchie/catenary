# Catenary

![http://dataphys.org/list/gaudis-hanging-chain-models/](http://dataphys.org/list/wp-content/uploads/2015/01/IMG_5393_640.jpg)

Catenary is a research codebase exploring an implementation of the ideas in
Henry Lin's [Bootstraps to Strings](https://arxiv.org/abs/2002.08387) paper.

(This README contains inline LaTeX! To view it, please use Chrome and install
[this Chrome
plugin](https://chrome.google.com/webstore/detail/mathjax-3-plugin-for-gith/peoghobgdhejhcmgoppjpjcidngdfkod).)

The code is organized as follows:

- The `mathematica/original` directory contains the original Mathematica code
  used to generate the figures in the paper, and some explorations from
  @sritchie.
- The rest of the notebooks in `mathematica` contain a library the generates the
  loop equations that we need to solve to solve matrix models.
- The `catenary` directory contains the Python code we've written to solve the
  single matrix model. One challenge will be extending this to search over
  systems with lots of parameters.

# Call for Collaboration

Here's a recipe for generating a new string theory:

1. [Generate a sequence of loop equations](#generate-the-loop-equations)
2. [Solve the loop equations, find a search space](#find-the-search-space)
3. [Assemble the variables into an 'inner product matrix'](#build-the-inner-product-matrix)
4. [Run gradient ascent on the min eigenvalue](#run-gradient-ascent)
5. [Find the critical points by tuning coupling constants](#find-the-critical-points)

We need your help with 2, 4 and 5. 4 and 5 are perfect for someone with
expertise in numerical methods, root-finding and equation solving.

We've built a [Mathematica
package](https://github.com/sritchie/catenary/blob/master/mathematica/MatrixModels.nb)
that neatly solves 1.
[MatrixModels.nb](https://github.com/sritchie/catenary/blob/master/mathematica/MatrixModels.nb)
has examples of how to generate loop equations for any arbitrary model.
[LoopEquations.wl](https://github.com/sritchie/catenary/blob/master/mathematica/LoopEquations.wl)
is a library you can import into other notebooks that implements all of this.

Mathematica can kind-of handle 2, but very slowly. There is so much structure in
the way that we generate the loop equations that we think it has to be possible
to exploit the structure and solve the equations much more quickly. This would
be a huge boost; the most loop equations we can solve, the higher our precision
can be as we search for the correct settings of our model's coupling constants.

We have an [implementation in
JAX](https://github.com/sritchie/catenary/blob/master/catenary/search.py#L95) of
4 that works well for a single search parameter, and should be able to be
extended without much work to the more general multiple-parameter case.

We also have an [implementation of
5](https://github.com/sritchie/catenary/blob/master/catenary/search.py#L189)
that works for a single coupling constant. The multi-matrix models we're
considering in the search for new physics need to search over two coupling
constants. Getting this working should be a small change, but making it work
fast and distributing it will be a challenge.

Please get in touch with me at [samritchie@x.team](mailto:samritchie@x.team) if
you'd like to help, would like a walkthrough of the code, or run into any
trouble here getting started. If you write me I'll get you in touch with other
folks who are interested, so we can divide and conquer.

## Background

You can describe a physical theory using a "matrix model", which is a
combination of some polynomial $V$ of (potentially many) matrix arguments:

$$V(A) = {1 \over 2} A^2 + {g \over 3} A^4$$

and some set of "symmetries". You can use this model to generate a series of
equations ("constraints") that we'll call the loop equations. These equations
look like this:

$$2 e_{\text{A}}+g e_{\text{AAAA}}+2 h e_{\text{AAAB}}+h e_{\text{AABB}}=e_{\text{AAA}}$$

How do you use this information to find new physics?

### Generate the Loop Equations

Take a matrix model and generate a large number of "loop equations". (This is
implemented in `mathematica/MatrixModels.nb`, so please see that notebook for a
detailed discussion of what this even means! For now just assume you have some
way to pump out an infinite number of these.)

[MatrixModels.nb](https://github.com/sritchie/catenary/blob/master/mathematica/MatrixModels.nb)
has examples of how to generate loop equations for any arbitrary model, and a
lot more background on what is happening here.

### Find the Search Space

The loop equations are full of variables like $e_{AABA}$. Each new equation
generates a constraint, and once you have enough loop equations, the number of
degrees of freedom settles down.

For the single matrix model described by

$$V(A) = {1 \over 2} A^2 + {g \over 3** A^4$$

There is in fact only a single search variable. If you specify $e_{AA}$, you can
use the loop equations to fill in every other variable.

You find the search space by asking a computer algebra system like Mathematica
to "solve" the equations for you, for all of the variables.

(There has to be a better way to do this. This is one area where we need help.)

### Build the inner product matrix

Next, you generate a big matrix with an interesting structure:

- Pick some ordering for your variables. Say, lexicographic ordering, so you
  might have, ignoring symmetries:

$$e_A, e_{B}, e_{AA}, e_{AB}, e_{BA}, e_{BB} ...$$

The goal is to assemble a square $N$ by $N$ matrix according to this rule:

- row 1 is the first $N$ elements of the list above
- row 2 is the the first row, with the first element's letters reversed and appended
- row 3 is the same, with element 2...

So with $N = 4$, you'd have this inner product matrix:

```
A,   B,   AA,   AB
AA,  AB,  AAA,  AAB
BA,  BB,  BAA,  BAB
AAA, AAB, AAAA, AAAB
```

Remember that these are all variables. The matrix is a function of the coupling
constants $g, k$ etc, and the "search variables" we found by solving the loop
equations.

We're interested in finding settings for the couplings where the matrix is still
JUST barely positive definite.

### Run gradient ascent

You can think of these as critical points of some function $f(g, k,...)$ of the
coupling constants. The function evaluates to the minimum eigenvalue of the
inner product matrix.

What are the settings for the search variables? We get to choose. It turns out
that we can run gradient *ascent* on the search variables, using the minimum
eigenvalue as the loss function. We follow the gradient and try to push the min
eigenvalue as high as it will go.

We have an [implementation in
JAX](https://github.com/sritchie/catenary/blob/master/catenary/search.py#L95) of
this function that works well for a single search variable, and should be able
to be extended without much work to the more general multiple-parameter case. *We
need your help here tuning the gradient ascent loop.

### Find the critical points

If we can find two settings for the inputs to $f(g, k,...)$ that give us a
positive and negative value, we can start honing in; we're looking for the point
where the minimum eigenvalue is 0, or just barely positive.

We also have a single-variable [implementation of this using bisection
search](https://github.com/sritchie/catenary/blob/master/catenary/search.py#L189)
that works for a single coupling constant. The multi-matrix models we're
considering in the search for new physics need to search over two coupling
constants. Getting this working should be a small change, but making it work
fast and distributing it will be a challenge. *We need your help!*

# Tools

This section describes how to install the various software tools that we use to
attack Catenary.

## Mathematica

[go/mathematica](http://go/mathematica) has instructions on how to get
Mathematica running on a corporate laptop or workstation.

## JAX / Python

[CONTRIBUTING.md](https://github.com/sritchie/catenary/blob/master/CONTRIBUTING.md)
has instructions on how to actually execute this code.

# Interested?

We would love your help! Please get in touch with me at
[samritchie@x.team](mailto:samritchie@x.team) and I'll add you to the private
Github repository where we're working on this.

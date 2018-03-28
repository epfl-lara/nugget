Nugget - Neural-Network Guided Expression Transformation
========================================================

The goal of this project is to explore the use of neural networks for term-rewriting.
Given two semantically equivalent expressions, we would like to transform the first expression
into the second, using only a fixed set of equality preserving transformations.
This gives an easily checkable proof that the two expressions are indeed equivalent.

We train a recursive neural network to help guide the search for a path between the expressions.
The network is trained to estimate the distance between expressions
(in terms of number of transformations),
as well as the most likely transformation to be applied.

The Expressions
---------------

In this project, we consider simple mathematical expressions composed of:

- Addition,
- Multiplication,
- Variables ``a``, ``b`` and ``c``,
- A single focus, which will be explained later on.

For instance, here are some expressions::

    [(a + b)]
    [(a + b)] * c
    (c * c) * (c * [a])

The Focus
^^^^^^^^^

The focus, denoted by square brackets (``[`` and ``]``) in the examples above,
indicates the point where transformations are to be applied.

The Transformations
-------------------

We support the following equality preserving transformations:

- Commutativity,
- Associativity (two directions),
- Distributivity (two directions).

All of those transformations are to be applied at the focus.

We also support the following three navigational transformations,
which move the focus around the expression:

- Focus up,
- Focus left,
- Focus right.

Example
-------

Below is an example of the kind of traces we would like to obtain.
In this case, we would like to find a path between the following two expressions::

    [(((a * a) * (b * a)) + (a * (a + b)))]

                      and

    [(((a * b) + (a * a)) + ((a * a) * (b * a)))]

One short path between the two expressions is::

    [(((a * a) * (b * a)) + (a * (a + b)))]

            =====  RIGHT  ====>

    (((a * a) * (b * a)) + [(a * (a + b))])

            == DISTRI_TIMES ==>

    (((a * a) * (b * a)) + [((a * a) + (a * b))])

            =====  COMMU  ====>

    (((a * a) * (b * a)) + [((a * b) + (a * a))])

            ======  UP  ======>

    [(((a * a) * (b * a)) + ((a * b) + (a * a)))]

            =====  COMMU  ====>

    [(((a * b) + (a * a)) + ((a * a) * (b * a)))]

How to Use
==========

Requirements
------------

- Python_ (version 2.7 or 3.x)
- PyTorch_ (version 0.3, available via pip)
- treenet_ (version 0.0.2, available via pip)

Install
-------

First of all, ensure that you have a Python_ installation::

    python --version

Then, clone the repository::

    git clone git@github.com:epfl-lara/nugget.git nugget
    cd nugget

If you have `Git LFS`_ installed, this should also fetch the data (this may take some time).
Otherwise, please see below for instructions on how to get the data, or even generate it yourself.

Next, install the python dependencies (PyTorch_ and treenet_). For instance, using ``pip``::

    pip install -r requirements.txt

.. _Python: https://www.python.org/
.. _PyTorch: http://pytorch.org
.. _treenet: https://github.com/epfl-lara/treenet

Data
----

The data used in this project has been generated from scratch.
It is available via `Git LFS`_, or by direct download from `our Github repository`_.

.. _Git LFS: https://git-lfs.github.com
.. _our Github repository: https://github.com/epfl-lara/nugget/tree/master/data

The data is of the following format::

    DISTANCE ; FIRST_EXPR ; SECOND_EXPR ; FIRST_TRANSFORMATION

Each record contains two expressions, the distance between the two expressions
(in terms of number of transformations),
as well as the first transformation applied on the path from the first expression to the second.
Expressions appear in prefix notation. For instance, here is such a record::

    7 ; * * * b a c C + + a b c ; * * b a * c + C + b c a ; ASSOC_LEFT

Generating the Data
^^^^^^^^^^^^^^^^^^^

To generate the data from scratch, you can use the following script::

    python -m nugget.generate

To see the available options for data generation::

    python -m nugget.generate --help

Training
--------

To train the neural network, issue the following command::

    python -m nugget.train

To see the available options for training::

    python -m nugget.train --help

FAQ
===


Why not Exhaustive Search ?
---------------------------

Trying to find a path between two expressions using an exhaustive search,
such as breadth-first search, quickly becomes impossible.
This is due to the fact that the number of states grows exponentially with the distance.

Why not use the Knuth-Bendix completion algorithm ?
---------------------------------------------------

Using the `Knuth-Bendix completion algorithm`_ would work in this very simple domain.
However, we would like to apply this technique to domains where the Knuth-Bendix completion is not applicable.

.. _Knuth-Bendix completion algorithm: https://en.wikipedia.org/wiki/Knuth-Bendix_completion_algorithm


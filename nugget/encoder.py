# Copyright 2018 EPFL.

import torch

from treenet import TreeEncoder

from nugget.expressions import NON_ATOM_ENTRIES, PLUS, TIMES

class ExpressionEncoder(TreeEncoder):

    def __init__(self, atoms):

        n = len(atoms)

        def value_fn(expr):
            value = torch.zeros(n + NON_ATOM_ENTRIES)
            if expr.is_binary():
                if expr.operator == PLUS:
                    value[0] = 1
                elif expr.operator == TIMES:
                    value[1] = 1
                else:
                    raise ValueError("Unknown operator: " + str(expr.operator))
            elif expr.is_focus():
                value[2] = 1
            elif expr.is_atom():
                index = atoms.index(expr.identifier) + NON_ATOM_ENTRIES
                value[index] = 1
            else:
                raise ValueError("Unknown expression: " + str(expr))
            return value

        def children_fn(expr):
            return expr.get_children()

        super(ExpressionEncoder, self).__init__(value_fn, children_fn)


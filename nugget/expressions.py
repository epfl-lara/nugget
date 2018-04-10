# Copyright 2018 EPFL.

# Binary operators.
TIMES = "*"
PLUS = "+"

# Focus marker.
FOCUS_MARKER = "C"

# Number of non-variable entries (+, * and focus).
NON_ATOM_ENTRIES = 3

# Transformations.
UP = "UP"
LEFT = "LEFT"
RIGHT = "RIGHT"
ASSOC_LEFT = "ASSOC_LEFT"
ASSOC_RIGHT = "ASSOC_RIGHT"
COMMU = "COMMU"
DISTRI_TIMES = "DISTRI_TIMES"
DISTRI_PLUS = "DISTRI_PLUS"

# Exhaustive list of transformations.
transformations = [
    UP,
    LEFT,
    RIGHT,
    ASSOC_LEFT,
    ASSOC_RIGHT,
    COMMU,
    DISTRI_TIMES,
    DISTRI_PLUS]

transformations_functions = {
    UP: lambda e: e.move_focus_up(),
    LEFT: lambda e: e.move_focus_left(),
    RIGHT: lambda e: e.move_focus_right(),
    ASSOC_LEFT: lambda e: e.apply_associativity_left(),
    ASSOC_RIGHT: lambda e: e.apply_associativity_right(),
    COMMU: lambda e: e.apply_commutativity(),
    DISTRI_TIMES: lambda e: e.apply_distributivity_times(),
    DISTRI_PLUS: lambda e: e.apply_distributivity_plus()
}


def from_prefix_notation(string):
    """Parse an expression from it's prefix notation.

    Args:
        string (string): A prefix notation expression.

    Returns:
        Expression: The parsed expression.
    """

    def parse(tokens):
        head = tokens[0]
        if head in [PLUS, TIMES]:
            (lhs, rest) = parse(tokens[1:])
            (rhs, remaining) = parse(rest)
            return (Binary(head, lhs, rhs), remaining)
        elif head == FOCUS_MARKER:
            (expr, rest) = parse(tokens[1:])
            return (Focus(expr), rest)
        else:
            return (Atom(head), tokens[1:])

    (parsed, rest) = parse(string.split())
    if rest:
        raise ValueError("Impossible to parse {}".format(str(string)))
    return parsed



class Expr(object):
    """Simple mathematical expression, possibly with a focus."""

    def __lt__(self, other):
        return False

    def to_prefix_notation(self):
        return ""

    def is_focus(self):
        return False

    def is_binary(self):
        return False

    def is_atom(self):
        return False

    def move_focus_up(self):
        return None

    def move_focus_left(self):
        return None

    def move_focus_right(self):
        return None

    def apply_commutativity(self):
        return None

    def apply_associativity_left(self):
        return None

    def apply_associativity_right(self):
        return None

    def apply_distributivity_times(self):
        return None

    def apply_distributivity_plus(self):
        return None

    def remove_focus(self):
        return self

    def get_focus(self):
        return None

    def get_children(self):
        return []

    def height(self):
        return None

    def length(self):
        return None


class Binary(Expr):
    """Binary operation."""

    def __init__(self, operator, lhs, rhs):
        self.operator = operator
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return "({} {} {})".format(self.lhs, self.operator, self.rhs)

    def __repr__(self):
        return "({} {} {})".format(self.lhs, self.operator, self.rhs)

    def to_prefix_notation(self):
        return "{} {} {}".format(self.operator,
            self.lhs.to_prefix_notation(), self.rhs.to_prefix_notation())

    def is_binary(self):
        return True

    def __eq__(self, that):
        if not isinstance(that, Binary):
            return False
        return (self.operator == that.operator and
                self.lhs == that.lhs and
                self.rhs == that.rhs)

    def __ne__(self, that):
        return not (self == that)

    def __hash__(self):
        return hash(("Binary", self.operator, self.lhs, self.rhs))

    def move_focus_up(self):
        if (self.lhs.is_focus()):
            return Focus(Binary(self.operator, self.lhs.expr, self.rhs))
        if (self.rhs.is_focus()):
            return Focus(Binary(self.operator, self.lhs, self.rhs.expr))

        lhs_up = self.lhs.move_focus_up()
        rhs_up = self.rhs.move_focus_up()

        if (lhs_up is not None and rhs_up is None):
            return Binary(self.operator, lhs_up, self.rhs)
        if (lhs_up is None and rhs_up is not None):
            return Binary(self.operator, self.lhs, rhs_up)

        return None

    def move_focus_left(self):

        lhs_left = self.lhs.move_focus_left()
        rhs_left = self.rhs.move_focus_left()

        if (lhs_left is not None and rhs_left is None):
            return Binary(self.operator, lhs_left, self.rhs)
        if (lhs_left is None and rhs_left is not None):
            return Binary(self.operator, self.lhs, rhs_left)

        return None

    def move_focus_right(self):

        lhs_right = self.lhs.move_focus_right()
        rhs_right = self.rhs.move_focus_right()

        if (lhs_right is not None and rhs_right is None):
            return Binary(self.operator, lhs_right, self.rhs)
        if (lhs_right is None and rhs_right is not None):
            return Binary(self.operator, self.lhs, rhs_right)

        return None

    def apply_commutativity(self):

        lhs_commu = self.lhs.apply_commutativity()
        rhs_commu = self.rhs.apply_commutativity()

        if (lhs_commu is not None and rhs_commu is None):
            return Binary(self.operator, lhs_commu, self.rhs)
        if (lhs_commu is None and rhs_commu is not None):
            return Binary(self.operator, self.lhs, rhs_commu)

        return None

    def apply_associativity_left(self):

        lhs_assoc = self.lhs.apply_associativity_left()
        rhs_assoc = self.rhs.apply_associativity_left()

        if (lhs_assoc is not None and rhs_assoc is None):
            return Binary(self.operator, lhs_assoc, self.rhs)
        if (lhs_assoc is None and rhs_assoc is not None):
            return Binary(self.operator, self.lhs, rhs_assoc)

        return None

    def apply_associativity_right(self):

        lhs_assoc = self.lhs.apply_associativity_right()
        rhs_assoc = self.rhs.apply_associativity_right()

        if (lhs_assoc is not None and rhs_assoc is None):
            return Binary(self.operator, lhs_assoc, self.rhs)
        if (lhs_assoc is None and rhs_assoc is not None):
            return Binary(self.operator, self.lhs, rhs_assoc)

        return None

    def apply_distributivity_times(self):
        lhs_distri = self.lhs.apply_distributivity_times()
        rhs_distri = self.rhs.apply_distributivity_times()

        if (lhs_distri is not None and rhs_distri is None):
            return Binary(self.operator, lhs_distri, self.rhs)
        if (lhs_distri is None and rhs_distri is not None):
            return Binary(self.operator, self.lhs, rhs_distri)

        return None

    def apply_distributivity_plus(self):
        lhs_distri = self.lhs.apply_distributivity_plus()
        rhs_distri = self.rhs.apply_distributivity_plus()

        if (lhs_distri is not None and rhs_distri is None):
            return Binary(self.operator, lhs_distri, self.rhs)
        if (lhs_distri is None and rhs_distri is not None):
            return Binary(self.operator, self.lhs, rhs_distri)

        return None

    def remove_focus(self):
        return Binary(self.operator,
            self.lhs.remove_focus(), self.rhs.remove_focus())

    def get_focus(self):
        lhs_focus = self.lhs.get_focus()
        rhs_focus = self.rhs.get_focus()

        if (lhs_focus is not None and rhs_focus is None):
            return lhs_focus
        if (lhs_focus is None and rhs_focus is not None):
            return rhs_focus

        return None

    def get_children(self):
        return [self.lhs, self.rhs]

    def height(self):
        return 1 + max(self.lhs.height(), self.rhs.height())

    def length(self):
        return 1 + self.lhs.length() + self.rhs.length()


class Atom(Expr):
    """Variable."""

    def __init__(self, identifier):
        self.identifier = identifier

    def __str__(self):
        return str(self.identifier)

    def __repr__(self):
        return str(self.identifier)

    def to_prefix_notation(self):
        return self.identifier

    def __eq__(self, other):
        if not isinstance(other, Atom):
            return False
        return self.identifier == other.identifier

    def __ne__(self, that):
        return not (self == that)

    def __hash__(self):
        return hash(("Atom", self.identifier))

    def is_atom(self):
        return True

    def height(self):
        return 0

    def length(self):
        return 1


class Focus(Expr):
    """Focused expression."""

    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return "[{}]".format(self.expr)

    def __repr__(self):
        return "[{}]".format(self.expr)

    def to_prefix_notation(self):
        return "{} {}".format(FOCUS_MARKER, self.expr.to_prefix_notation())

    def __eq__(self, other):
        if not isinstance(other, Focus):
            return False
        return self.expr == other.expr

    def __ne__(self, that):
        return not (self == that)

    def __hash__(self):
        return hash((FOCUS_MARKER, self.expr))

    def is_focus(self):
        return True

    def move_focus_left(self):

        if not self.expr.is_binary():
            return None

        return Binary(self.expr.operator, Focus(self.expr.lhs), self.expr.rhs)

    def move_focus_right(self):

        if not self.expr.is_binary():
            return None

        return Binary(self.expr.operator, self.expr.lhs, Focus(self.expr.rhs))

    def apply_commutativity(self):

        if not self.expr.is_binary():
            return None

        return Focus(Binary(self.expr.operator, self.expr.rhs, self.expr.lhs))

    def apply_associativity_left(self):

        if not (self.expr.is_binary() and
                self.expr.lhs.is_binary() and
                self.expr.operator == self.expr.lhs.operator):
            return None

        return Focus(Binary(
            self.expr.operator,
            self.expr.lhs.lhs,
            Binary(self.expr.operator, self.expr.lhs.rhs, self.expr.rhs)))

    def apply_associativity_right(self):

        if not (self.expr.is_binary() and
                self.expr.rhs.is_binary() and
                self.expr.operator == self.expr.rhs.operator):
            return None

        return Focus(Binary(
            self.expr.operator,
            Binary(self.expr.operator, self.expr.lhs, self.expr.rhs.lhs),
            self.expr.rhs.rhs))

    def apply_distributivity_times(self):
        if not (self.expr.is_binary() and
                self.expr.rhs.is_binary() and
                self.expr.operator == TIMES and
                self.expr.rhs.operator == PLUS):
            return None
        a = self.expr.lhs
        b = self.expr.rhs.lhs
        c = self.expr.rhs.rhs
        return Focus(Binary(PLUS, Binary(TIMES, a, b), Binary(TIMES, a, c)))

    def apply_distributivity_plus(self):
        if not (self.expr.is_binary() and
                self.expr.lhs.is_binary() and
                self.expr.rhs.is_binary() and
                self.expr.operator == PLUS and
                self.expr.lhs.operator == TIMES and
                self.expr.rhs.operator == TIMES and
                self.expr.lhs.lhs == self.expr.rhs.lhs):
            return None

        a = self.expr.lhs.lhs
        b = self.expr.lhs.rhs
        c = self.expr.rhs.rhs
        return Focus(Binary(TIMES, a, Binary(PLUS, b, c)))

    def remove_focus(self):
        return self.expr

    def get_focus(self):
        return self

    def get_children(self):
        return [self.expr]

    def height(self):
        return 1 + self.expr.height()

    def length(self):
        return 1 + self.expr.length()


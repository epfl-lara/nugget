# Copyright 2018 EPFL.

import argparse
import random

from nugget.expressions import *


def random_expr(depth, atoms):
    """Return a random expression, with a Focus.

    Args:
        depth: Desired depth of the expression.
        atoms: Available atoms.
    """

    a = random_expr_without_focus(depth, atoms)
    (p, _) = random_downwards_moves_until_atom(Focus(a), None)
    return random.choice(p)


def random_expr_without_focus(depth, atoms):
    """Return a random expression, without a Focus.

    Args:
        depth: Desired depth of the expression.
        atoms: Available atoms.
    """

    if depth <= 0:
        return Atom(random.choice(atoms))

    lower = random.randint(0, depth - 1)
    upper = depth - 1

    (left, right) = (lower, upper)\
        if random.choice([True, False]) else (upper, lower)

    return Binary(
        random.choice([PLUS, TIMES]),
        random_expr_without_focus(left, atoms),
        random_expr_without_focus(right, atoms))


def random_distri_times_expr(depth, atoms):
    """Return a random expression, with a Focus.

    The returned expression is garanteed to support
    distributivity (first direction).

    Args:
        depth: Desired depth of the expression.
        atoms: Available atoms.
    """

    depth = depth - 2

    depth_focus = random.randint(0, depth)
    remaining_depth = depth - depth_focus

    depth_repeated = random.randint(0, remaining_depth)

    lower = random.randint(0, remaining_depth)
    upper = remaining_depth

    (depth_left, depth_right) = (lower, upper)\
        if random.choice([True, False]) else (upper, lower)

    repeated = random_expr_without_focus(depth_repeated, atoms)
    left = random_expr_without_focus(depth_left, atoms)
    right = random_expr_without_focus(depth_right, atoms)
    inner = Binary(TIMES, repeated, Binary(PLUS, left, right))
    return random_context(inner, depth, depth_focus, atoms)


def random_distri_plus_expr(depth, atoms):
    """Return a random expression, with a Focus.

    The returned expression is garanteed to support
    distributivity (second direction).

    Args:
        depth: Desired depth of the expression.
        atoms: Available atoms.
    """

    depth = depth - 2

    depth_focus = random.randint(0, depth)
    remaining_depth = depth - depth_focus

    depth_repeated = random.randint(0, remaining_depth)

    lower = random.randint(0, remaining_depth)
    upper = remaining_depth

    (depth_left, depth_right) = (lower, upper)\
        if random.choice([True, False]) else (upper, lower)

    repeated = random_expr_without_focus(depth_repeated, atoms)
    left = random_expr_without_focus(depth_left, atoms)
    right = random_expr_without_focus(depth_right, atoms)
    inner = Binary(
        PLUS,
        Binary(TIMES, repeated, left),
        Binary(TIMES, repeated, right))
    return random_context(inner, depth, depth_focus, atoms)


def random_context(expr, depth_total, depth_focus, atoms):
    """Generate a random context.

    Args:
        expr: The expression to be focused.
        depth_total: Desired depth of the returned expression.
        depth_focus: Desired depth of the focus.
        atoms: Available atoms.
    """

    if depth_focus <= 0:
        return Focus(expr)

    focus_in_left = random.choice([True, False])
    depth_without = random.randint(0, depth_total)
    with_focus = random_context(expr, depth_total - 1, depth_focus - 1, atoms)
    without_focus = random_expr_without_focus(depth_without, atoms)
    operator = random.choice([PLUS, TIMES])

    if focus_in_left:
        return Binary(operator, with_focus, without_focus)
    else:
        return Binary(operator, without_focus, with_focus)


def random_transformations(start, length):
    """Apply random transformations on an expression.

    Args:
        start: The starting expression.
        length: Number of transformations to be applied.
    """

    seen = {}
    focus = []
    current = start
    remaining = length
    seen[current] = remaining
    restarts = 10
    while remaining > 0:
        # Up phase.
        nUp = random.randint(0, min(remaining - 1, len(focus)))
        k = len(focus) - nUp
        last = focus[k] if nUp > 0 else None
        focus = focus[:k]
        for _ in range(nUp):
            current = current.move_focus_up()
            previous_remaining = seen.get(current, None)
            if (previous_remaining is not None and
                    remaining < previous_remaining):
                remaining = previous_remaining
                if restarts <= 0:
                    return None
                else:
                    restarts -= 1
            else:
                remaining -= 1
                seen[current] = remaining

        # Down phase.
        (exprs, dirs) = random_downwards_moves_until_atom(current, last)
        nDown = random.randint(0, min(remaining - 1, len(exprs) - 1))
        for i in range(nDown):
            current = exprs[1 + i]
            previous_remaining = seen.get(current, None)
            if (previous_remaining is not None and
                    remaining < previous_remaining):
                remaining = previous_remaining
                if restarts <= 0:
                    return None
                else:
                    restarts -= 1
            else:
                remaining -= 1
                seen[current] = remaining
        focus.extend(dirs[:nDown])

        # Apply transformation.
        neighbors = [current.apply_commutativity(),
                     current.apply_associativity_left(),
                     current.apply_associativity_right(),
                     current.apply_distributivity_times(),
                     current.apply_distributivity_plus()]
        neighbors = [n for n in neighbors if n is not None]
        current = random.choice(neighbors)

        previous_remaining = seen.get(current, None)
        if previous_remaining is not None and remaining < previous_remaining:
            remaining = previous_remaining
            if restarts <= 0:
                return None
            else:
                restarts -= 1
        else:
            remaining -= 1
            seen[current] = remaining

    return current


def random_downwards_moves_until_atom(expr, prohibit_first=None):
    """Apply LEFT or RIGHT moves until an Atom is reached.

    Args:
        expr: The expression, with a Focus.
        prohibit_first: Move to prohibit for the first move.
            Can be None, LEFT or RIGHT. Default: None.

    Returns:
        All expressions on the path and the list of applied moves.
    """

    path = []
    directions = []
    expr_focus = expr.get_focus().expr
    while expr_focus.is_binary():
        path.append(expr)
        if random.choice([LEFT, RIGHT]) == LEFT and prohibit_first != LEFT:
            directions.append(LEFT)
            expr = expr.move_focus_left()
            expr_focus = expr_focus.lhs
        else:
            directions.append(RIGHT)
            expr = expr.move_focus_right()
            expr_focus = expr_focus.rhs
        prohibit_first = None
    return (path, directions)


def random_pair(depth, atoms, max_distance):
    """Return two random expressions, each with a Focus.

    The returned expressions are guaranteed to be reachable
    from one another by at most max_distance transformations.

    Args:
        depth: Desired depth of the expression.
        atoms: Available atoms.
        max_distance: Maximum distance between the two expressions.
    """

    a = random_expr(depth, atoms)
    b = random_transformations(a, max_distance)
    while b is None:
        max_distance -= 1
        b = random_transformations(a, max_distance)
    return (a, b)


def explore(from_expr, max_depth, including=None):
    """Perform a breadth-first exploration.

    Args:
        from_expr: Starting expression.
        max_depth: Maximum distance to explore.
        including: First transformation that must be part of the output.
            The exploration will be aborted early if the first transformation
            does not lead to deep enough branches.

    Returns:
        A dictionary, index by transformations, of sample expressions reachable
        using the key as first transformation.
    """

    seen = {from_expr: 0}
    queue = [(from_expr, 0, None)]
    previous_depth = 0
    buffers = {}
    outputs = {}

    for t in transformations:
        buffers[t] = []
        outputs[t] = []

    while queue:
        (current_expr, current_depth, current_branch) = queue.pop()
        if current_depth != previous_depth:
            if including is not None and not buffers[including]:
                return outputs
            for (k, es) in buffers.items():
                if es:
                    outputs[k].append((previous_depth + 1, random.choice(es)))
            buffers = {}
            for t in transformations:
                buffers[t] = []
            previous_depth = current_depth

        next_depth = current_depth + 1

        ts = transformations[:]
        random.shuffle(ts)
        for t in ts:
            next_expr = transformations_functions[t](current_expr)
            if (next_expr is not None and
                    ((not next_expr in seen) or seen[next_expr] == next_depth)):
                seen[next_expr] = next_depth
                branch = current_branch if current_branch is not None else t
                buffers[branch].append(next_expr)
                if next_depth < max_depth:
                    queue.insert(0, (next_expr, next_depth, branch))

    for (k, es) in buffers.items():
        if es:
            outputs[k].append((previous_depth + 1, random.choice(es)))

    return outputs


DEFAULT_COUNT = 1000
DEFAULT_MIN_DEPTH = 2
DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_DISTANCE = 10
DEFAULT_ATOMS = list("abc")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate data.")
    parser.add_argument("-c", "--count", type=int,
        help="number of examples per class and distance",
        default=DEFAULT_COUNT)
    parser.add_argument("-m", "--max-distance", type=int,
        help="maximum distance between the expressions",
        default=DEFAULT_MAX_DISTANCE)
    parser.add_argument("--min-depth", type=int,
        help="minimum depth of the expressions",
        default=DEFAULT_MIN_DEPTH)
    parser.add_argument("--max-depth", type=int,
        help="maximum depth of the expressions",
        default=DEFAULT_MAX_DEPTH)
    parser.add_argument("-a", "--atoms", nargs="+", type=str,
        help="atoms",
        default=DEFAULT_ATOMS)
    args = parser.parse_args()

    count = args.count
    max_distance = args.max_distance
    atoms = args.atoms
    min_depth = args.min_depth
    max_depth = args.max_depth

    counts = {}
    for t in transformations:
        counts[t] = 0

    while True:
        min_count = None
        min_action = None
        for t in transformations:
            c = counts[t]
            if min_count is None or c < min_count:
                min_count = c
                min_action = t

        if min_count == count:
            break

        picked_depth = random.randint(min_depth, max_depth)
        if min_action == DISTRI_TIMES:
            a = random_distri_times_expr(max(picked_depth, 2), atoms)
        elif min_action == DISTRI_PLUS:
            a = random_distri_plus_expr(max(picked_depth, 2), atoms)
        else:
            a = random_expr(picked_depth, atoms)
        steps = explore(a, max_distance, min_action)
        for (first_action, elements) in steps.items():
            c = counts[first_action]
            if ((min_count is None or c == min_count) and
                    c < count and len(elements) == max_distance):
                counts[first_action] = c + 1
                for (distance, b) in elements:
                    print("{} ; {} ; {} ; {}".format(
                        distance,
                        a.to_prefix_notation(),
                        b.to_prefix_notation(),
                        first_action))


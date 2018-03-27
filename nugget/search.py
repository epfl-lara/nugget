# Copyright 2018 EPFL.

from heapq import *
import random

from nugget.expressions import *

def history_to_csv(history):
    lines = []
    lines.append(','.join(['id','expr','estimatedDistance','action','parentId']))
    for (expr_id, expr, estimated_distance, action, parent_id) in history:
        lines.append(','.join([
            str(expr_id),
            str(expr),
            str(estimated_distance) if estimated_distance is not None else '',
            action if action is not None else '',
            str(parent_id) if parent_id is not None else '']))
    return '\n'.join(lines)

def best_first_search(from_expr, to_expr, heuristics, factor=0.0):
    if from_expr == to_expr:
        return ([from_expr], [], [])

    h = heuristics.with_target(to_expr)

    # Building the entry of the first expression.
    (d, ts) = h(from_expr)
    ts = list(zip(ts, range(0, len(transformations))))
    ts.sort()
    ts = [t for (_, t) in ts]

    parents = { from_expr: (None, None) }
    to_visit = [(d, from_expr, 0, ts)]

    # For logging purposes.
    ids = { from_expr: 0 }
    history = [(0, from_expr, d, None, None)]
    next_id = 1

    while to_visit:
        (current_estimated_distance, current_expr, current_depth, current_children) = to_visit[0]
        if current_children:
            transformation = transformations[current_children.pop()]
            next_children = transformations_functions[transformation](current_expr)
            next_depth = current_depth + 1

            if next_children is not None:
                if next_children not in parents:  # Checking that the expr was not already visited.
                    parents[next_children] = (current_expr, transformation)

                    (d, ts) = h(next_children)

                    history.append((next_id, next_children, d, transformation, ids[current_expr]))
                    ids[next_children] = next_id
                    next_id += 1

                    if next_children == to_expr:  # Checking if we reached the target expression.
                        break

                    ts = list(zip(ts, range(0, len(transformations))))
                    ts.sort()
                    ts = [t for (_, t) in ts]

                    heappush(to_visit, (d + next_depth * factor, next_children, next_depth, ts))
        else:
            heappop(to_visit)

    path = [to_expr]
    actions = []
    current_expr = to_expr
    while current_expr is not None:
        (current_expr, action) = parents[current_expr]
        if current_expr is not None:
            path.insert(0, current_expr)
            actions.insert(0, action)

    return (path, actions, history)


def breadth_first_search(from_expr, to_expr):
    if from_expr == to_expr:
        return ([from_expr], [], [])

    parents = { from_expr: (None, None) }
    queue = [from_expr]

    # For logging purposes.
    ids = { from_expr: 0 }
    history = [(0, from_expr, None, None, None)]
    next_id = 1

    while queue:
        current_expr = queue.pop()

        for transformation in transformations:
            next_expr = transformations_functions[transformation](current_expr)
            if next_expr is not None and not next_expr in parents:
                parents[next_expr] = (current_expr, transformation)

                ids[next_expr] = next_id
                history.append((next_id, next_expr, None, transformation, ids[current_expr]))
                next_id += 1
                queue.insert(0, next_expr)

                if next_expr == to_expr:
                    path = [to_expr]
                    actions = []
                    current_expr = to_expr
                    while current_expr is not None:
                        (current_expr, action) = parents[current_expr]
                        if current_expr is not None:
                            path.insert(0, current_expr)
                            actions.insert(0, action)

                    return (path, actions, history)


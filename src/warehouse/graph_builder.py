
import math
import re

import networkx as nx


def euclidean_distance(point_a, point_b):
    """
    Calculate distance between two points on the same floor.
    """
    if point_a["Z"] != point_b["Z"]:
        raise ValueError(
            "Cannot calculate a walking edge between different floors."
        )

    return math.hypot(
        point_a["X"] - point_b["X"],
        point_a["Y"] - point_b["Y"],
    )


def parse_support_label(label):
    """
    Parse labels such as CC-08 and LC-08.
    Returns (prefix, number), or None if invalid.
    """
    match = re.fullmatch(
        r"\s*(CC|LC)-(\d+)\s*",
        str(label).upper()
    )

    if not match:
        return None

    return match.group(1), int(match.group(2))


def build_warehouse_graph(support_points):
    """
    Build a navigation graph from support points.

    Rules:
    1. CC and LC with the same number are connected horizontally.
    2. Consecutive CC points on the same vertical aisle are connected
       only when their coordinates confirm vertical alignment.

    Important:
    CC-to-CC edges are conservative candidates. Confirm the actual
    physical aisle connectivity before using this graph for routing.
    """
    graph = nx.Graph()

    # Add support points as graph nodes
    for _, row in support_points.iterrows():
        label = str(row["labels"]).strip().upper()

        graph.add_node(
            label,
            x=float(row["X"]),
            y=float(row["Y"]),
            z=float(row["Z"]),
        )

    # Index nodes by label
    nodes_by_label = {}

    for node, attrs in graph.nodes(data=True):
        parsed = parse_support_label(node)

        if parsed is None:
            continue

        prefix, number = parsed

        nodes_by_label.setdefault(
            (prefix, number), []
        ).append(node)

    # Connect CC-n to LC-n when their labels share a number.
    for (prefix, number), cc_nodes in nodes_by_label.items():
        if prefix != "CC":
            continue

        lc_nodes = nodes_by_label.get(("LC", number), [])

        for cc in cc_nodes:
            for lc in lc_nodes:
                cc_attrs = graph.nodes[cc]
                lc_attrs = graph.nodes[lc]

                # Horizontal connection must be on the same floor
                # and aligned on the same Y coordinate.
                if (
                    cc_attrs["z"] == lc_attrs["z"]
                    and math.isclose(
                        cc_attrs["y"],
                        lc_attrs["y"],
                        abs_tol=1e-6,
                    )
                ):
                    distance = euclidean_distance(
                        cc_attrs, lc_attrs
                    )

                    graph.add_edge(
                        cc,
                        lc,
                        weight=distance,
                        edge_type="horizontal",
                    )

    # Candidate vertical CC connections.
    # Connect consecutive CC numbers only when they share X and Z.
    cc_nodes = [
        node
        for node in graph.nodes
        if parse_support_label(node)
        and parse_support_label(node)[0] == "CC"
    ]

    for cc in cc_nodes:
        _, number = parse_support_label(cc)
        next_nodes = nodes_by_label.get(("CC", number + 1), [])

        for next_cc in next_nodes:
            a = graph.nodes[cc]
            b = graph.nodes[next_cc]

            if (
                a["z"] == b["z"]
                and math.isclose(a["x"], b["x"], abs_tol=1e-6)
                and not math.isclose(a["y"], b["y"], abs_tol=1e-6)
            ):
                distance = euclidean_distance(a, b)

                graph.add_edge(
                    cc,
                    next_cc,
                    weight=distance,
                    edge_type="vertical_candidate",
                )

    return graph

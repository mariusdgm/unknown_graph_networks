import numpy as np
import networkx as nx


def graph_sanity(A):
    A = np.asarray(A, dtype=float)
    diag_max = float(np.max(np.abs(np.diag(A))))
    rs = A.sum(axis=1)
    asym = float(np.linalg.norm(A - A.T) / (np.linalg.norm(A) + 1e-12))
    edges = int(np.sum(A > 1e-12))

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    sccs = list(nx.strongly_connected_components(G))
    sink_sizes = []
    for S in sccs:
        is_sink = True
        for u in S:
            for v in G.successors(u):
                if v not in S:
                    is_sink = False
                    break
            if not is_sink:
                break
        if is_sink:
            sink_sizes.append(len(S))

    return dict(
        diag_max=diag_max,
        row_sum_min=float(rs.min()),
        row_sum_mean=float(rs.mean()),
        row_sum_max=float(rs.max()),
        asym=asym,
        edges=edges,
        sink_sizes=sink_sizes,
        has_singleton_sink=any(sz == 1 for sz in sink_sizes),
    )

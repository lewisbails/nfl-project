from scipy.spatial.distance import mahalanobis, euclidean
import networkx as nx
from networkx.algorithms.matching import max_weight_matching as mwm
import numpy as np
import pandas as pd
from typing import Union


def match(data: pd.DataFrame, t: str, distance: str, method: str, caliper: Union[int, float] = 5):
    ''' Returns mapping of treatment group member to its control group match.

    Parameters
    ----------
    data: DataFrame
        Consists of examples from both treatment and control groups.
    t: str
        Treatment variable (column in data)
    distance: str
        Distance measure between examples.
    method: str
        Matching method for pairing control examples to treatment variables.
    caliper: int, float
        Maximum distance between examples of a proposed match.

    Returns
    -------
    matches : dict

    '''

    treatment = data.loc[data[t], :].index  # treatment players
    control = data.loc[~data[t], :].index  # control players
    # dont include distances between like-group members
    distances = get_distances(data.drop(t, axis=1), distance).loc[control, treatment.values]

    if method == 'approximate':
        matches = approximate_matches(distances, caliper)
    if method == 'graphical':
        matches = graphical_matches(distances, caliper)

    return matches


def approximate_matches(distance_matrix, threshold):
    # sort the treatment neightbours of each control example
    distances = {c_idx: list(filter(lambda x: x[-1] <= threshold, sorted(
        zip(row.index, row), key=lambda x: x[-1]))) for c_idx, row in distance_matrix.iterrows()}
    # take a step forward for each control, noting the treatment examples removed from the available pool every time
    # this isnt optimal but it might be an okay approximation
    matches = {}
    no_match = set()
    step = 1
    while len(matches) + len(no_match) < len(distance_matrix.index):
        print(f'Step {step}')
        new_distances = {}
        for c_idx, remaining in distances.items():
            if len(remaining):
                t_idx, dist = remaining[0]
                if t_idx not in matches:
                    # add match
                    print(f'Match {c_idx} -> {t_idx} ({round(dist,1)})')
                    matches[t_idx] = {'c': c_idx, 'dist': dist}
                else:
                    # already matched, step forward instead
                    control_exists = matches[t_idx]['c']
                    print(
                        f'{c_idx} -> {t_idx} ({round(dist,1)}) already taken by {control_exists} -> {t_idx}, {len(remaining[1:])} options remaining.')
                    new_distances[c_idx] = remaining[1:]
            else:
                # no treatments available to match (i.e. the rest were further than caliper value away)
                print(f'No more available matches for {c_idx}.')
                no_match.add(c_idx)
        distances = new_distances
        step += 1
    return matches


def get_distances(u: pd.DataFrame, method: str):
    distances = {}
    if method == 'mahalanobis':
        inv_cov = np.linalg.inv(u.cov())
        for i, row_i in u.iterrows():
            distances_i = [mahalanobis(row_i, row_j, inv_cov) for _, row_j in u.iterrows()]
            distances[i] = pd.Series(distances_i, index=u.index)
    elif method == 'euclidean':
        for i, row_i in u.iterrows():
            distances_i = [euclidean(row_i, row_j) for _, row_j in u.iterrows()]
            distances[i] = pd.Series(distances_i, index=u.index)
    distance_matrix = pd.DataFrame.from_dict(distances, orient='index')
    return distance_matrix


def get_graph(u: pd.DataFrame, c):
    graph = nx.Graph()
    for c_idx, row in u.iterrows():
        for t_idx, dist in row.items():
            if dist <= c:
                graph.add_edge(t_idx, c_idx, weight=-dist)  # negative distances for minimum weight matching
    return graph


def graphical_matches(data: pd.DataFrame, threshold):
    graph = get_graph(data, threshold)
    matches = mwm(graph, maxcardinality=True)
    # NOTE: key is not guaranteed to be the treatment example in the match
    return {k: {'c': v, 'dist': -graph.get_edge_data(k, v)['weight']} for k, v in matches}

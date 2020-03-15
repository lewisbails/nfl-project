from scipy.spatial.distance import pdist, squareform
import networkx as nx
from networkx.algorithms.matching import max_weight_matching as mwm
import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns


def filter_indices(distances, method: str, caliper: Union[int, float] = 5):
    if method == 'with_replacement':
        if caliper == 'auto':
            caliper = distances.std().mean()
            print(f'Using caliper {caliper}')
        matches = distances.le(caliper)
    else:
        raise NotImplementedError('Matching method not found.')
    # controls that have a treatment match
    control_idx = [idx for idx in matches.index if matches.loc[idx, :].sum()]
    # treatments that have a control match
    treatment_idx = [idx for idx in matches.columns if matches[idx].sum()]
    return control_idx + treatment_idx


def get_distances(data, distance, t):
    treatment = data.loc[data[t], :].index  # treatment players
    control = data.loc[~data[t], :].index  # control players
    distances = pdist(data.drop(t, axis=1).astype(float), distance)
    distances = pd.DataFrame._from_arrays(squareform(distances), index=data.index, columns=data.index.values)
    distances = distances.loc[control, treatment.values]
    return distances


def match(data, on, dv, distance, method, caliper):
    distances = get_distances(data.drop(dv, axis=1), distance, on)
    df_matched = data.loc[filter_indices(distances, method, caliper), :]
    return df_matched

# def match_with_replacement(distance_matrix, threshold):

#     matches = {}
#     if more_control:
#         iterator = distance_matrix.items()  # iterate over treatment
#     else:
#         iterator = distance_matrix.iterrows()  # iterate over control

#     for l_idx, row in iterator:
#         if row.min() < threshold:
#             matches[l_idx] = {'match': row.idxmin(), 'dist': row.min()}
#     return matches


# def approximate_matches(distance_matrix, threshold):
#     # sort the treatment neightbours of each control example
#     distances = {c_idx: list(filter(lambda x: x[-1] <= threshold, sorted(
#         zip(row.index, row), key=lambda x: x[-1]))) for c_idx, row in distance_matrix.iterrows()}
#     # take a step forward for each control, noting the treatment examples removed from the available pool every time
#     # this isnt optimal but it might be an okay approximation
#     matches = {}
#     no_match = set()
#     step = 1
#     while len(matches) + len(no_match) < len(distance_matrix.index):
#         print(f'Step {step}')
#         new_distances = {}
#         for c_idx, remaining in distances.items():
#             if len(remaining):
#                 t_idx, dist = remaining[0]
#                 if t_idx not in matches:
#                     # add match
#                     print(f'Match {c_idx} -> {t_idx} ({round(dist,1)})')
#                     matches[t_idx] = {'match': c_idx, 'dist': dist}
#                 else:
#                     # already matched, step forward instead
#                     control_exists = matches[t_idx]['match']
#                     print(
#                         f'{c_idx} -> {t_idx} ({round(dist,1)}) already taken by {control_exists} -> {t_idx}, {len(remaining[1:])} options remaining.')
#                     new_distances[c_idx] = remaining[1:]
#             else:
#                 # no treatments available to match (i.e. the rest were further than caliper value away)
#                 print(f'No more available matches for {c_idx}.')
#                 no_match.add(c_idx)
#         distances = new_distances
#         step += 1
#     return matches


# def get_graph(u: pd.DataFrame, c):
#     graph = nx.Graph()
#     for c_idx, row in u.iterrows():
#         for t_idx, dist in row.items():
#             if dist <= c:
#                 graph.add_edge(t_idx, c_idx, weight=-dist)  # negative distances for minimum weight matching
#     return graph


# def graphical_matches(data: pd.DataFrame, threshold):
#     graph = get_graph(data, threshold)
#     matches = mwm(graph, maxcardinality=True)
#     # NOTE: key is not guaranteed to be the treatment example in the match
#     return {k: {'match': v, 'dist': -graph.get_edge_data(k, v)['weight']} for k, v in matches}


def covariate_dists(data, on, **kwargs):
    vals = data[on].unique()
    for col in data.drop(on, axis=1).columns:
        try:
            for val in vals:
                sns.distplot(data[data[on] == val][col], label=f'{on}={val}', hist=False)
        except:
            for val in vals:
                sns.distplot(data[data[on] == val][col],
                             label=f'{on}={val}', hist=True, kde=False, norm_hist=True)
        plt.title(f'{col} distributions')
        plt.legend()
        plt.show()


def dist_test(data, on, func, **kwargs):

    v = data[on].unique()
    if len(v) != 2:
        print('Dichotamize variable of interest and try again.')
        return None

    res = []
    for col in data.drop(on, axis=1).columns:
        if len(data[col].unique()) <= 2:
            res.append([np.nan])
        else:
            res.append(func(data.loc[:, [on, col]], on))

    return pd.DataFrame.from_records(res, index=data.drop(on, axis=1).columns, columns=['statistic'])


def binom_z_tests(data, on):
    from scipy.stats import norm

    v = data[on].unique()
    if len(v) != 2:
        print(f'Dichotamize variable of interest and try again. {len(v)}')
        return None

    res = []
    for col in data.drop(on, axis=1).columns:
        if len(data[col].unique()) != 2:
            res.append([np.nan])
        else:
            p1 = data[data[on] == v[0]][col].mean()
            p2 = data[data[on] == v[1]][col].mean()
            n1 = len(data[data[on] == v[0]][col])
            n2 = len(data[data[on] == v[1]][col])
            p = (n1 * p1 + n2 * p2) / (n1 + n2)
            z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
            res.append([z, norm.sf(abs(z)) * 2])

    return pd.DataFrame.from_records(res, index=data.drop(on, axis=1).columns, columns=['statistic'])


def search_radii(data, on, dv, distance, method, radii, cont_test):
    radii_res = []
    for radius in radii:
        distances = get_distances(data.drop(dv, axis=1), distance, on)
        df_matched = data.loc[filter_indices(distances, method, radius), :]
        if len(df_matched):
            # Imbalance measure 1: mean min mahalanobis distance
            min_dist = distances.min()
            mean_min = min_dist[min_dist < radius].mean()
            # Imbalance measure 2: histogram bin frequence
            # TODO: implement
            # Imbalance measure 3: difference of means?
            # TODO: implement
            # Imbalance measure 4: Dist of propensity scores?
            # TODO: implement
# BROKEN

            # mw = dist_test(df_matched.drop(dv, axis=1), on=on, func=cont_test)
            # bt = binom_z_tests(df_matched.drop(dv, axis=1), on=on)
            # res = mw['statistic'].combine_first(bt['statistic'])
            # res.index.rename('covariate', inplace=True)
            # res = res.reset_index()
            # res['treatments'] = df_matched[on].sum()
            # res['samples'] = len(df_matched)
            # res['radius'] = radius
        else:
            res = pd.DataFrame({'statistic': [np.nan] * (len(df_matched.columns) - 2),
                                'covariate': df_matched.drop([dv, on], axis=1).columns})
            res['treatments'] = 0
            res['samples'] = 0
            res['radius'] = radius
        radii_res.append(res)

    cov_dists = pd.concat(radii_res)
    cov_dists
    return cov_dists


def mahalanobis_discrepancy(data, on):
    treatment = data.loc[data[on], :].index  # treatment
    control = data.loc[~data[on], :].index  # control
    # dont include distances between like-group members
    distances = pdist(data.drop(on, axis=1).astype(float), 'mahalanobis')
    distances = pd.DataFrame._from_arrays(squareform(distances), index=data.index, columns=data.index.values)
    distances = distances.loc[control, treatment.values]

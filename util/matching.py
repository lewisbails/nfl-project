from scipy.spatial.distance import pdist, squareform
import networkx as nx
from networkx.algorithms.matching import max_weight_matching as mwm
import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns


def mahalanobis_frontier(data, on, dv):
    distances = get_distances(data.drop(dv, axis=1), 'mahalanobis', on)
    MMD_tc = distances.min(axis=0)  # gives closest control to each treatment
    MMD_ct = distances.min(axis=1)  # gives closest treatment to each control
    MMD = pd.concat([MMD_tc, MMD_ct])

    pruned_controls = []
    pruned_treatments = []
    AMD_ = []
    radii = []
    unique_distances = sorted(set(np.ravel(distances.round(1).to_numpy())), reverse=True)
    print(f'{len(unique_distances)} distances to check.')
    for i, radius in enumerate(unique_distances):
        if not i % 10:
            print(f'Checked {(i+1)}/{len(unique_distances)} distances')
        AMD = MMD[MMD <= radius].mean()  # average min distance below threshold
        AMD_.append(AMD)
        treatments, controls = prune_by_distance(distances, radius)  # controls that werent pruned
        pruned_controls.append(len(distances) - len(controls))
        pruned_treatments.append(len(distances.columns) - len(treatments))
        radii.append(radius)

        # stopping criterion
        if pruned_controls[-1] == len(distances) or AMD_[-1] == 0:
            break

    return pd.DataFrame.from_dict({'pruned controls': pruned_controls, 'pruned treatments': pruned_treatments, 'AMD': AMD_, 'radius': radii}, orient='columns')


def L_frontier(data, on, dv):
    data = data.drop(dv, axis=1)
    controls = len(data[~data[on]])
    treatments = len(data[data[on]])

    # find number of bins to use
    n_L1 = L1(data, on)
    H = n_L1[n_L1 == n_L1.median()].index[0]

    remaining = data.copy()
    pruned_controls = []
    L1s = []
    while len(remaining):
        # assess
        L1s.append(L1(remaining, on, n_bins=[H]).median())
        pruned_controls.append(len(controls) - len(remaining[~remaining[on]]))
        pruned_treatments.append(len(treatments) - len(remaining[~remaining[on]]))

        # stopping criterion
        if pruned_controls[-1] == controls or pruned_treatments == treatments or L1s[-1] == 0:
            break

        # prune
        remaining = prune_by_hist(remaining, on)

    return pd.DataFrame.from_records([pruned_controls, pruned_treatments, L1s], columns=['pruned controls', 'pruned treatments', 'L1'])


def prune_by_hist(data, on):
    treatment, control = list(map(lambda x: x.drop(on, axis=1), split(data, on)))
    ranges = [(data[col].min(), data[col].max()) for col in treatment.columns]
    t_histd, edges = np.histogramdd(treatment, H, range=ranges, density=True)
    c_histd, _ = np.histogramdd(control, H, range=ranges, density=True)
    t_hist = np.array(np.histogramdd(treatment, H, range=ranges, density=False)[0], dtype=bool)
    c_hist = np.array(np.histogramdd(control, H, range=ranges, density=False)[0], dtype=bool)
    d_hist = (t_histd - c_histd) * t_hist * c_dist  # dont consider bins of only T or C
    idx_absmax = np.unravel_index(np.argmax(abs(d_hist), axis=None), d_hist.shape)
    left_edge = edges[idx_absmax]
    right_edge = edges[(i + 1 for i in idx_absmax)]
    if d_hist[idx_absmax] > 0:
        treatment = drop_one_in_bin(treatment, left_edge, right_edge)
    else:
        control = drop_one_in_bin(control, left_edge, right_edge)
    data = data.loc[treatment.index + control.index, :]
    return data


def drop_one_in_bin(data, left_edge, right_edge):
    mask = [all(fi >= left_edge) and all(fi < right_edge) for fi in data.to_array()]
    to_drop = data.loc[mask, :].sample(1).index[0]
    data = data.drop(index=to_drop)
    return data


def prune_by_distance(distances, caliper: Union[int, float] = 5):
    matches = distances.le(caliper)
    # controls that have a treatment match
    control_idx = [idx for idx in matches.index if matches.loc[idx, :].sum()]
    # treatments that have a control match
    treatment_idx = [idx for idx in matches.columns if matches[idx].sum()]
    return treatment_idx, control_idx


def get_distances(data, distance, t):
    treatment, control = list(map(lambda x: x.index, split(data, t)))
    distances = pdist(data.drop(t, axis=1).astype(float), distance)
    distances = pd.DataFrame._from_arrays(squareform(distances), index=data.index, columns=data.index.values)
    distances = distances.loc[control, treatment.values]
    distances.index.rename('control', inplace=True)
    distances.columns.rename('treatment', inplace=True)
    return distances


def match_by_distance(data, on, dv, distance, caliper):

    if caliper == 'auto':
        df = mahalanobis_frontier(data, on, dv)
        df['AMD'] = (df['AMD'] - df['AMD'].min()) / (df['AMD'].max() - df['AMD'].min())
        df['pruned controls'] = (df['pruned controls'] - df['pruned controls'].min()) / \
            (df['pruned controls'].max() - df['pruned controls'].min())
        df['dto'] = np.sqrt(df['AMD']**2 + df['pruned controls']**2)
        caliper = df.loc[df['dto'] == df['dto'].min(), 'radius']

    distances = get_distances(data.drop(dv, axis=1), distance, on)
    treatments, controls = prune_by_distance(distances, caliper)
    df_matched = data.loc[controls + treatments, :]
    return df_matched


def match_by_L1(data, on, dv, l1):
    pass


def Ln(data, on, func, **kwargs):
    treatment, control = list(map(lambda x: x.drop(on, axis=1), split(data, on)))
    n_bins = np.arange(5, 20) if n_bins not in kwargs else kwargs['n_bins']
    ranges = [(data[col].min(), data[col].max()) for col in treatment.columns]
    L1_s = []
    for bins in n_bins:
        t_hist, _ = np.histogramdd(treatment, bins, range=ranges, density=True)
        c_hist, _ = np.histogramdd(control, bins, range=ranges, density=True)
        L1_s.append(func(t_hist, c_hist))
    res = pd.Series(L1_s, index=n_bins, name='distance')
    res.index.rename('bin width', inplace=True)
    return res


def L1(data, on, **kwargs):
    return Ln(data, on, lambda x, y: np.sum(np.abs(x - y)) / 2, **kwargs)


def L2(data, on, **kwargs):
    return Ln(data, on, lambda x, y: np.sqrt(np.sum((x - y)**2) / 2), **kwargs)


def dom(data, on, func):
    norm = (data - data.mean()) / data.std()
    treatment, control = list(map(lambda x: x.drop(on, axis=1), split(norm, on)))
    return func(treatment.mean() - control.mean())


def dom_2(data, on):
    return dom(data, on, lambda x: np.sqrt(np.sum(x**2) / len(x)))


def dom_1(data, on):
    return dom(data, on, lambda x: np.sum(x.abs()) / len(x))


def split(data, on):
    return data.loc[data[on], :], data.loc[~data[on], :],


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
#     distances = {c_idx: list(prune(lambda x: x[-1] <= threshold, sorted(
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

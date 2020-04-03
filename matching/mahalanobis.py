from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd


def mahalanobis_frontier(data, on):
    print('Distances..', end=' ')
    distances = get_distances(data, 'mahalanobis', on)
    MMD_tc = distances.min(axis=0)  # gives closest control to each treatment
    MMD_ct = distances.min(axis=1)  # gives closest treatment to each control
    MMD = pd.concat([MMD_tc, MMD_ct])
    print('ready.')

    pruned_controls = []
    pruned_treatments = []
    AMD_ = []
    radii = []
    unique_distances = sorted(set(np.ravel(distances.round(1).to_numpy())), reverse=True)
    for radius in tqdm(unique_distances):
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


def prune_by_distance(distances, caliper: Union[int, float] = 5):
    matches = distances.le(caliper)
    # controls that have a treatment match
    control_idx = [idx for idx in matches.index if matches.loc[idx, :].sum()]
    # treatments that have a control match
    treatment_idx = [idx for idx in matches.columns if matches[idx].sum()]
    return treatment_idx, control_idx


def get_distances(data, distance, t, dtype=float):
    treatment, control = list(map(lambda x: x.index, split(data, t)))
    distances = pdist(data.drop(t, axis=1).astype(dtype), distance)
    distances = pd.DataFrame._from_arrays(squareform(distances), index=data.index, columns=data.index.values)
    distances = distances.loc[control, treatment.values]
    distances.index.rename('control', inplace=True)
    distances.columns.rename('treatment', inplace=True)
    return distances


def match_by_distance(data, on, dv, distance, caliper):
    if caliper == 'auto':
        df = mahalanobis_frontier(data.drop(dv, axis=1), on)
        df['AMD'] = (df['AMD'] - df['AMD'].min()) / (df['AMD'].max() - df['AMD'].min())
        df['pruned controls'] = (df['pruned controls'] - df['pruned controls'].min()) / \
            (df['pruned controls'].max() - df['pruned controls'].min())
        df['dto'] = np.sqrt(df['AMD']**2 + df['pruned controls']**2)
        caliper = df.loc[df['dto'] == df['dto'].min(), 'radius']

    distances = get_distances(data.drop(dv, axis=1), distance, on)
    treatments, controls = prune_by_distance(distances, caliper)
    df_matched = data.loc[controls + treatments, :]
    return df_matched


def split(data, on):
    return data.loc[data[on], :], data.loc[~data[on], :]

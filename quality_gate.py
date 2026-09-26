"""Versioned, narrow input gates for frozen HARTH summary inference."""
import numpy as np
from robustness import infer

POLICIES = ('finite-v1', 'exact-zero-v1')


def input_reasons(features, policy):
    if policy not in POLICIES:
        raise ValueError('unknown input policy')
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != 30:
        raise ValueError('expected N by 30 features')
    reasons = np.full(len(x), '', dtype='<U24')
    finite = np.isfinite(x).all(axis=1)
    if policy == 'exact-zero-v1':
        back = (x[:, :15] == 0).all(axis=1)
        thigh = (x[:, 15:] == 0).all(axis=1)
        reasons[finite & back] = 'zero_back_sensor'
        reasons[finite & thigh] = 'zero_thigh_sensor'
        reasons[finite & back & thigh] = 'zero_both_sensors'
    reasons[~finite] = 'missing_feature'
    return reasons


def gated_infer(model, features, threshold, policy):
    reasons = input_reasons(features, policy)
    gated = np.asarray(features, dtype=float).copy()
    gated[reasons != ''] = np.nan
    return infer(model, gated, threshold), reasons

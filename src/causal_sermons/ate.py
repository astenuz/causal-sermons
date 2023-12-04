import numpy as np


def tau_naive(t, y):
    return y[t == 1].mean(axis=0) - y[t == 0].mean(axis=0)


def tau_PI(q_t0, q_t1):
    ite = (q_t1 - q_t0)
    return ite.mean(axis=0)


def tau_IPW(g, t, y):
    ite = (t / g - (1 - t) / (1 - g)) * y
    return ite.mean(axis=0)


def tau_DR(q_t0, q_t1, g, t, y):
    full_q = q_t0 * (1 - t) + q_t1 * t
    h = t * (1.0 / g) - (1.0 - t) / (1.0 - g)
    ite = h * (y - full_q) + q_t1 - q_t0

    return ite.mean(axis=0)


def all_ate_estimators(q_t0, q_t1, g, t, y):
    return {
        'tau_naive': tau_naive(t, y),
        'tau_PI': tau_PI(q_t0, q_t1),
        'tau_IPW': tau_IPW(g, t, y),
        'tau_DR': tau_DR(q_t0, q_t1, g, t, y),
    }
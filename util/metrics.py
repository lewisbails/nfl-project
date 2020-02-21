import numpy as np


def tetrachoric(c1, c2):
    c11 = c1 == 1
    c10 = c1 == 0
    c21 = c2 == 1
    c20 = c2 == 0

    a = sum(c11 * c20)
    b = sum(c11 * c21)
    c = sum(c10 * c20)
    d = sum(c10 * c21)
    corr = np.cos(np.radians(180 / (1 + np.sqrt(b * c / a / d))))
    return corr

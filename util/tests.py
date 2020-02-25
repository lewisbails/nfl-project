from scipy.stats import chi2


def LR_test(l1, l2, dof=1):
    LR = -2 * (l1) - (-2 * (l2))
    p = chi2.sf(LR, dof)
    return p

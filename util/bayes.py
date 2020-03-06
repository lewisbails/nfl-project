from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np


class BetaBinomial:
    def __init__(self, prior_alpha, prior_beta, name):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.posterior_alpha = prior_alpha
        self.posterior_beta = prior_beta
        self.name = name

    def plot_prior(self, x=None, show=False, **kwargs):
        if x is None:
            x = np.linspace(0, 1, 500)
        plt.plot(x, beta(self.prior_alpha, self.prior_beta).pdf(x), label='prior distribution', **kwargs)
        if show:
            plt.xlabel('p')
            plt.ylabel('density')
            plt.title('Prior distributions for Field goal success rate.')
            plt.show()

    def plot_posterior(self, x=None, show=False, prior=True, **kwargs):
        if x is None:
            x = np.linspace(0, 1, 500)
        if prior:
            self.plot_prior(x, **{'linestyle': 'dashed', 'linewidth': 0.5})
        plt.plot(x, beta(self.posterior_alpha, self.posterior_beta).pdf(
            x), label=f'posterior of {self.name}', **kwargs)
        if show:
            plt.xlabel('p')
            plt.ylabel('density')
            plt.title('Posterior distributions for Field goal success rate.')
            plt.legend()
            plt.show()

    def observe(self, observations):
        self.posterior_alpha += observations.sum()
        self.posterior_beta += len(observations) - observations.sum()
        return self

    def sample(self, x=20000):
        return beta(self.posterior_alpha, self.posterior_beta).rvs(x)

import pystan
import pandas as pd 
import numpy as np 
from numpy.random import *
import seaborn as sns
import matplotlib.pyplot as plt
from plotter.regression_alalysis_plots import *
from scipy.stats import mstats

"""

"""

def main():
    np.random.seed(23)

    N = 10
    X = np.arange(N)
    Y = 13.436 + 2.718 * X + np.random.standard_cauchy(N)

    data = {
        'N': N,
        'X': X,
        'Y': Y,
    }

    code = """
    data {
        int N;
        real X[N];
        real Y[N];
    }

    parameters {
        real b1;
        real b2;
        real<lower=0> sigma;
    }

    model {
        for (n in 1:N)
            Y[n] ~ cauchy(b1 + b2 * X[n], sigma);
    }
    
    generated quantities {
        real y_pred[N];
        for (n in 1:N)
            y_pred[n] = cauchy_rng(b1 + b2 * X[n], sigma);
    }
    """
    
    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=2)
    la = fit.extract(permuted=True)
    print(fit)
    
    b1 = np.mean(la['b1'])
    b2 = np.mean(la['b2'])
    y_pred = la['y_pred']
    low_y, high_y = mstats.mquantiles(y_pred, [0.025, 0.975], axis=0)
    plt.fill_between(X, low_y, high_y, alpha=0.3, color="gray")
    plt.plot(X, Y, "o")
    plt.plot(X, b1 + b2*X)

    plt.show()
    fit.plot()


if __name__ == '__main__':
    main()
import pystan
from numpy.random import *
import matplotlib.pyplot as plt

"""
単回帰分析
"""

def main():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    data = {
        'N': len(df),
        'X': list(df['sepal length (cm)']),
        'Y': list(iris.target)
    }

    code = """
    data {
        int N;
        real Y[N];
    }

    parameters {
        real mu;
        real sig;
    }

    model {
        for (n in 1:N) {
            Y[n] ~ normal(mu, sig);
        }
    }
    """
    
    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=2)
    la = fit.extract(permuted=True)  # return a dictionary of arrays
    print(la['mu'][-1])
    print(la['sig'][-1])
    fit.plot()
    plt.show()

    # schools_code = """
    # data {
    #     int<lower=0> J; // number of schools
    #     vector[J] y; // estimated treatment effects
    #     vector<lower=0>[J] sigma; // s.e. of effect estimates
    # }
    # parameters {
    #     real mu;
    #     real<lower=0> tau;
    #     vector[J] eta;
    # }
    # transformed parameters {
    #     vector[J] theta;
    #     theta = mu + tau * eta;
    # }
    # model {
    #     eta ~ normal(0, 1);
    #     y ~ normal(theta, sigma);
    # }
    # """

    # schools_dat = {'J': 8,
    #             'y': [28,  8, -3,  7, -1,  1, 18, 12],
    #             'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

    # sm = pystan.StanModel(model_code=schools_code)
    # fit = sm.sampling(data=schools_dat, iter=1000, chains=2)


if __name__ == '__main__':
    main()
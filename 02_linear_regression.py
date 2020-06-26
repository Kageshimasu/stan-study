import pystan
from sklearn.datasets import load_iris
from stan_plotter.regression_alalysis_plots import *

"""
単回帰分析
"""


def main():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = pd.DataFrame(iris.target)
    data = {
        'N': len(df),
        'X': list(df['sepal length (cm)']),
        'Y': list(iris.target)
    }

    code = """
    data {
        int N;
        real Y[N];
        real X[N];
    }

    parameters {
        real b1;
        real b2;
        real<lower=0> sig;
    }
    
    transformed parameters {
        real mu[N];
        for (n in 1:N)
            mu[n] = b1 + b2*X[n];
    }

    model {
        for (n in 1:N)
            Y[n] ~ normal(mu[n], sig);
    }

    generated quantities {
        real y_pred[N];
        for (n in 1:N)
            y_pred[n] = normal_rng(mu[n], sig);
    }
    """

    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=2)
    la = fit.extract(permuted=True)
    print(fit)
    print('b1: {}'.format(np.mean(la['b1'])))
    print('b2: {}'.format(np.mean(la['b2'])))
    fit.plot()

    plot_predicted_vs_observed(df['target'], la['y_pred'])
    plot_noise_distribution(df['target'], la['mu'])
    plt.show()


if __name__ == '__main__':
    main()

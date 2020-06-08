import pystan
from numpy.random import *
import matplotlib.pyplot as plt


def main():
    N1 = 30
    N2 = 20
    data = {
        'N1': N1,
        'N2': N2,
        'Y1': list(normal(0, 5, N1)),
        'Y2': list(normal(1, 4, N2))
    }

    code = """
    data {
        int N1;
        int N2;
        real Y1[N1];
        real Y2[N2];
    }

    parameters {
        real mu1;
        real mu2;
        real<lower=0> sigma;
    }

    model {
        for (n in 1:N1)
            Y1[n] ~ normal(mu1, sigma);
        for (n in 1:N2)
            Y2[n] ~ normal(mu2, sigma);
    }
    """
    iter_n = 1000
    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=iter_n, chains=2)
    la = fit.extract(permuted=True)  
    prob = sum([1 for i in range(iter_n) if la['mu1'][i] < la['mu2'][i]]) / iter_n
    print(prob)
    fit.plot()
    plt.show()


if __name__ == '__main__':
    main()
import pystan
from numpy.random import *
import matplotlib.pyplot as plt

"""
パラメータμ、σの正規分布から生成されたあるデータから、
パラメータμ、σを推定する
"""

def main():
    # 平均50、分散10の正規分布から100個の乱数を生成する
    N = 100
    data = {
        'N': N,
        'Y': list(normal(50, 10, N))
    }

    # stanのコード
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
    la = fit.extract(permuted=True)
    print(la['mu'][-1])
    print(la['sig'][-1])
    fit.plot()
    plt.show()


if __name__ == '__main__':
    main()

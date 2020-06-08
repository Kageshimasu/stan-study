import pystan
import pandas as pd
import numpy as np
from numpy.random import *
import seaborn as sns
import matplotlib.pyplot as plt
from plotter.regression_alalysis_plots import *
from scipy.stats import mstats

"""
階層モデル
階層モデルは、個体差の種別(カテゴリー)が判明している状況で、
個体差間を考慮したモデル。階層的に事前分布を設定してるため
階層モデルと呼ぶ。
m1 = バイアスの全体平均
m2 = 傾きの全体平均
b1[K] ~ N(m1, sigma_b1) b1は全体平均を通してばらついている -> すべてのパラメータは同じ分布から生成されたと仮定
b2[K] ~ N(m2, sigma_b2) b2は全体平均を通してばらついている -> すべてのパラメータは同じ分布から生成されたと仮定
Y[N] ~ N(b1[k] + b2[k] * X[n], sigma) 年収は各パラメータから生成
"""


def main():
    df = pd.read_csv('./data/data_salary.csv')
    Y = df['Y'].tolist()  # 年収
    X = df['X'].tolist()  # 年齢
    K = df['KID']  # どの会社に属するか

    data = {
        'N': len(df),
        'Y': Y,
        'X': X,
        'K': K.max(),
        'KID': K,
    }

    code = """
    data {
        int N;
        int K;
        real X[N];
        real Y[N];
        int KID[N];
    }

    parameters {
        real b1[K];
        real b2[K];
        real m1;
        real m2;
        real<lower=0> sigma_b1;
        real<lower=0> sigma_b2;
        real<lower=0> sigma;
    }

    model {
        for (k in 1:K) {
            b1[k] ~ normal(m1, sigma_b1);
            b2[k] ~ normal(m2, sigma_b2);
        }
    
        for (n in 1:N) {
            Y[n] ~ normal(b1[KID[n]] + b2[KID[n]] * X[n], sigma);
        }
    }
    
    generated quantities {
        real y_pred[N];
        for (n in 1:N)
            y_pred[n] = normal_rng(b1[KID[n]] + b2[KID[n]] * X[n], sigma);
    }
    """

    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=2)
    la = fit.extract(permuted=True)
    print(fit)
    print(np.mean(la['y_pred'], axis=0))

    plt.show()
    fit.plot()


if __name__ == '__main__':
    main()

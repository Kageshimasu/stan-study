import pystan
import pandas as pd 
import numpy as np 
from numpy.random import *
import seaborn as sns
import matplotlib.pyplot as plt
from plotter.regression_alalysis_plots import *

"""
ターゲットが0or1なのでベルヌーイ分布を過程できる。
よってロジスティック回帰をおこなう。
"""


def main():
    ## データ読み込み
    df = pd.read_csv('./data/data_attendance_for_logistic.csv')
    part_time = df['like_part-time_job'].tolist()
    interest_in_learning = (df['interest_in_learning'] / 200).tolist()
    weather = df['weather'].tolist()
    for i, w in enumerate(weather):
        if w == 'A':
            weather[i] = 0
        elif w == 'B':
            weather[i] = 0.2
        else:
            weather[i] = 1
    attend_a_class = df['attend_a_class'].tolist()

    ## PyStan用にdictにデータ格納
    data = {
        'N': len(attend_a_class),
        'part_time': part_time,
        'interest_in_learning': interest_in_learning,
        'weather': weather,
        'attend_a_class': attend_a_class
    }

    ## PyStanのコード記入
    code = """
    data {
        int N;
        int<lower=0, upper=1> part_time[N];
        real<lower=0, upper=1> interest_in_learning[N];
        real<lower=0, upper=1> weather[N];
        int<lower=0> attend_a_class[N];
    }

    parameters {
        real b1;
        real b2;
        real b3;
        real b4;
    }

    transformed parameters {
        real<lower=0, upper=1> q[N];

        for (n in 1:N)
            q[n] = inv_logit(b1 + b2*part_time[n] + b3*interest_in_learning[n] + b4*weather[n]);
    }

    model {
        for (n in 1:N)
            attend_a_class[n] ~ bernoulli(q[n]);
    }
    
    generated quantities {
        real y_pred[N];
        for (n in 1:N)
            y_pred[n] = bernoulli_rng(q[n]);
    }
    """
    
    ## MCMC実行
    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=2)

    ## MCMC結果抽出
    la = fit.extract(permuted=True)
    plot_predicted_vs_observed(df['attend_a_class'], la['y_pred'])
    fit.plot()
    plt.show()


if __name__ == '__main__':
    main()
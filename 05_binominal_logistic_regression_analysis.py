import pystan
import pandas as pd 
import numpy as np 
from numpy.random import *
import seaborn as sns
import matplotlib.pyplot as plt
from plotter.regression_alalysis_plots import *

"""
3カ月の授業回数分、何回出席したのかを推定する。
ターゲットの確率変数が0以上の整数で上限が決まっているため、二項分布として過程できる。
この場合二項ロジスティック回帰が有効となる。
残差プロットは未実装だが、plot_predicted_vs_observedで精度がわかる。
"""

def main():
    df = pd.read_csv('./data/data_attendance_for_bin.csv')
    A = df['like_part-time_job'].tolist()
    Score = (df['interest_in_learning'] / 200).tolist()
    M = df['num_classes_for_three_months'].tolist()
    Y = df['num_of_attendances_for_the_three_months'].tolist()
    data = {
        'N': len(Y),
        'A': A,
        'Score': Score,
        'M': M,
        'Y': Y
    }

    code = """
    data {
        int N;
        int<lower=0, upper=1> A[N];
        real<lower=0, upper=1> Score[N];
        int<lower=0> M[N];
        int<lower=0> Y[N];
    }

    parameters {
        real b1;
        real b2;
        real b3;
    }

    transformed parameters {
        real<lower=0, upper=1> q[N];

        for (n in 1:N)
            q[n] = inv_logit(b1 + b2*A[n] + b3*Score[n]);
    }

    model {
        for (n in 1:N)
            Y[n] ~ binomial(M[n], q[n]);
    }
    
    generated quantities {
        real y_pred[N];
        for (n in 1:N)
            y_pred[n] = binomial_rng(M[n], q[n]);
    }
    """
    
    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=2)
    la = fit.extract(permuted=True)
    plot_predicted_vs_observed(df['num_of_attendances_for_the_three_months'], la['y_pred'])
    fit.plot()
    plt.show()


if __name__ == '__main__':
    main()
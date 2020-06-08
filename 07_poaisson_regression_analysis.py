import pystan
import pandas as pd
import numpy as np
from numpy.random import *
import seaborn as sns
import matplotlib.pyplot as plt
from plotter.regression_alalysis_plots import *

"""
3カ月の授業の出席回数をターゲットとして予測する。
授業回数はカウントデータなため、非負の値をとる。
非負のカウントデータはポアソン回帰が有効である。
lambdaは平均授業回数を表す
Scoreが150点の時のlambda / Scoreが50点の時のlambda = 授業回数が平均で~倍変化するといえる
"""


def main():
    df = pd.read_csv('./data/data_attendance_for_bin.csv')
    A = df['like_part-time_job'].tolist()
    Score = (df['interest_in_learning'] / 200).tolist()
    M = df['num_classes_for_three_months'].tolist()
    data = {
        'N': len(df),
        'A': A,
        'Score': Score,
        'M': M,
    }

    code = """
    data {
        int N;
        int<lower=0, upper=1> A[N];
        real<lower=0, upper=1> Score[N];
        int<lower=0> M[N];
    }

    parameters {
        real b[3];
    }

    transformed parameters {
        real<lower=0> lambda[N];

        for (n in 1:N)
            lambda[n] = exp(b[1] + b[2] * A[n] + b[3] * Score[n]);
    }

    model {
        for (n in 1:N)
            M[n] ~ poisson(lambda[n]);
    }
    
    generated quantities {
        real y_pred[N];
        for (n in 1:N)
            y_pred[n] = poisson_rng(lambda[n]);
    }
    """

    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=2)
    la = fit.extract(permuted=True)
    print(fit)
    # plot_predicted_vs_observed(df['num_classes_for_three_months'], la['y_pred'])
    fit.plot()
    plt.show()


if __name__ == '__main__':
    main()

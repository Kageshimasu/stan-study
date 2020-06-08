import pystan
import pandas as pd 
import numpy as np 
from numpy.random import *
import seaborn as sns
import matplotlib.pyplot as plt
from plotter.regression_alalysis_plots import *

"""
重回帰分析
バイト好き、学習への興味が一年の出席率にどの程度影響するか
plot_predicted_vs_observed:予想した値と観測値の外れ具合。45度線上に点が打たれるといい。
plot_noise_distribution:残差をプロット。正規分布のようになるはず。
"""

def main():
    df = pd.read_csv('./data/data_attendance_for_mult.csv')
    A = df['like_part-time_job'].tolist()
    Score = df['interest_in_learning'].tolist()
    Y = df['attendance_rate_for_a_year'].tolist()
    data = {
        'N': len(Y),
        'A': A,
        'Score': Score,
        'Y': Y
    }

    code = """
    data {
        int N;
        real A[N];
        real Score[N];
        real Y[N];
    }

    parameters {
        real b1;
        real b2;
        real b3;
        real<lower=0> sigma;
    }

    transformed parameters {
        real mu[N];
        for (n in 1:N)
            mu[n] = b1 + b2*A[n] + b3*Score[n] / 200;
    }

    model {
        for (n in 1:N)
            Y[n] ~ normal(mu[n], sigma);
    }
    
    generated quantities {
        real y_pred[N];
        for (n in 1:N)
            y_pred[n] = normal_rng(mu[n], sigma);
    }
    """
    
    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=2)
    la = fit.extract(permuted=True)
    print(fit)
    print(df['attendance_rate_for_a_year'])
    print(la['y_pred'].shape)
    fit.plot()
    
    plot_predicted_vs_observed(df['attendance_rate_for_a_year'], la['y_pred'])
    plot_noise_distribution(df['attendance_rate_for_a_year'], la['mu'])
    plt.show()


if __name__ == '__main__':
    main()
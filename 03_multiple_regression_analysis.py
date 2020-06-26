import pystan
from stan_plotter.regression_alalysis_plots import *

"""
重回帰分析
「バイト好き」、「学習への興味」が1年の出席率にどの程度影響するか
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
    fit.plot()

    plot_predicted_vs_observed(df['attendance_rate_for_a_year'], la['y_pred'])
    plot_noise_distribution(df['attendance_rate_for_a_year'], la['mu'])
    plt.show()


if __name__ == '__main__':
    main()

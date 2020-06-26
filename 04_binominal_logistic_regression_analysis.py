import pystan
from stan_plotter.regression_alalysis_plots import *

"""
二項ロジスティック回帰分析
■背景
3カ月の授業のうち何回出席したのかを推定する。
ターゲットの確率変数が0以上の整数で上限が決まっているため、二項分布として過程できる。
この場合二項ロジスティック回帰が有効となる。
plot_predicted_vs_observedで当てはまりを調べる。

■技術要素
尤度を示す。
N: データ数(Observed)
n_i: i番目のマックスのデータ数(Observed)
x_i: i番目の確率変数、0~n_iの整数(Observed)
b_i: i番目の回帰係数
a_i[n]: n個目のデータにおけるi番目の説明変数

p_i = logistic_function(b_0 + b_1*a_1[i] + , ... )
L(b | N, n, p, x) = Π_{i=1}^{N} n_iCx_i p_i^{x_i} (1-p_i)^{n_i-x_i}
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
    print(fit)
    plot_predicted_vs_observed(df['num_of_attendances_for_the_three_months'], la['y_pred'])
    fit.plot()
    plt.show()


if __name__ == '__main__':
    main()
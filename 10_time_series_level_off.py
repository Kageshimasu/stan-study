import pystan
import arviz
from stan_plotter.regression_alalysis_plots import *

"""
時系列データの頭打ち
患者に点滴で薬剤を投与する場合を考える。
投与からの経過時間Timeと薬の血中濃度Yのデータに対して回帰分析を行う場合のtips。

a: 頭打ちの大きさを決める
b: 頭打ちになるまでの時間
y = a*(1 - exp(-b*T[n]))

S字型の増え方なら： C / {1 + a*exp(-bt)}
ある時から減少するなら： C*{exp(-b_2*t) - exp(-b_1*t)}
"""


def main():
    df = pd.read_csv('./data/data_time_series.csv')
    t = df['Time']
    y = df['Y']

    data = {
        'N': len(df),
        'Y': y,
        'T': t,
        'MaxT': t.max()
    }
    code = """
    data {
        int N;
        int MaxT;
        real Y[N];
        real T[N];
    }

    parameters {
        real<lower=0, upper=100> a;
        real<lower=0, upper=5> b;
        real<lower=0> s;
    }

    model {
        for (n in 1:N)
            Y[n] ~ normal(a*(1 - exp(-b*T[n])), s);
    }

    generated quantities {
        real y_pred[MaxT];
        for (n in 1:MaxT)
            y_pred[n] = normal_rng(a*(1 - exp(-b*n)), s);
    }
    """

    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=800, chains=3)
    la = fit.extract(permuted=True)
    a = np.mean(la['a'])
    b = np.mean(la['b'])
    print(fit)
    print('parameter a: {}'.format(a))
    print('parameter b: {}'.format(b))

    f = lambda t: a * (1 - np.exp(-b * t))
    x = np.arange(1, t.max() + 1, 1)
    plot_linear_regression(df['Time'], df['Y'], la['y_pred'], x, f(x))
    arviz.plot_trace(fit)
    plt.show()


if __name__ == '__main__':
    main()

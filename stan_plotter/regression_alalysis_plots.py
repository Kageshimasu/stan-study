import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_predicted_vs_observed(y_true: pd.DataFrame, y_pred: np.ndarray):
    """
    予想した値と観測値の外れ具合を表す。
    横軸が観測値に対して、縦軸が予測結果の4分位範囲を示した図。
    ばらつきがなく、45度線上に点が打たれるほどいい。
    :param y_true:　真の値
    :param y_pred:　予測結果
    :return:
    """
    quantile = [10, 50, 90]
    quantile_colname = ['p' + str(x) for x in quantile]
    df_pred_quantile = pd.DataFrame(np.percentile(y_pred, q=quantile, axis=0).T, columns=quantile_colname)
    df = pd.concat([y_true, df_pred_quantile], axis=1)

    palette = sns.color_palette()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    min_ax = np.min(y_true) - np.mean(y_true) * 0.1
    max_ax = np.max(y_true) + np.mean(y_true) * 0.1
    ax.plot([min_ax, max_ax], [min_ax, max_ax], 'k--', alpha=0.7)
    ax.errorbar(
        df.iloc[:, 0],
        df.p50,
        yerr=[df.p50 - df.p10, df.p90 - df.p50],
        fmt='o',
        ecolor='gray',
        ms=10,
        mfc=palette[0],
        alpha=0.8,
        marker='o')
    ax.set_aspect('equal')
    ax.set_xlim(min_ax, max_ax)
    ax.set_ylim(min_ax, max_ax)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')


def plot_noise_distribution(y_true: pd.DataFrame, mu_pred: np.ndarray, estimator=np.mean):
    """
    残差のヒストグラムを表す。
    正規分布に近いほどうまく当てはまっている
    :param y_true: 真の値
    :param mu_pred: 予測結果
    :param estimator: default 平均値
    :return:
    """
    df_noises = y_true - estimator(mu_pred, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    sns.distplot(list(df_noises), ax=ax)


def plot_linear_regression(x_true, y_true, y_pred, x, y):
    """
    回帰上に50%区間、90%区間を描画する
    :param y_true:
    :param x_true:
    :param y_pred:
    :param x:
    :param y:
    :return:
    """
    quantile = [5, 25, 75, 95]
    quantile_colname = ['p' + str(x) for x in quantile]
    df_pred_quantile = pd.DataFrame(np.percentile(y_pred, q=quantile, axis=0).T, columns=quantile_colname)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x_true, y_true)
    # ax.plot(x, y)
    ax.fill_between(x, df_pred_quantile.p25, df_pred_quantile.p75, color='#262626', alpha=0.4)
    ax.fill_between(x, df_pred_quantile.p5, df_pred_quantile.p95, color='#888888', alpha=0.4)

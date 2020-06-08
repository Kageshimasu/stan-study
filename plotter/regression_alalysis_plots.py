import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_predicted_vs_observed(y_true, y_pred):
    quantile = [10, 50, 90]
    quantile_colname = ['p' + str(x) for x in quantile]
    df_pred_quantile = pd.DataFrame(np.percentile(y_pred, q=quantile, axis=0).T, columns=quantile_colname)
    df = pd.concat([y_true, df_pred_quantile], axis=1)

    palette = sns.color_palette()
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
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


def plot_noise_distribution(y_true, mu_pred, estimator=np.mean):
    df_noises = y_true - estimator(mu_pred, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    sns.distplot(list(df_noises), ax=ax)


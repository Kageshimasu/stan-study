import pystan
from stan_plotter.regression_alalysis_plots import *

"""
Latent Dirichlet Allocation
離散的な値のクラスタリングに使用可能。
今回の分析の目的は、購入履歴から顧客の特徴を抽出したい、商品をグルーピングしたいとする。

■どのようにデータが生成されているのか仮定の説明。
1. ある顧客はある割合Θで、商品のグループKを選択する。
2. 商品のグループKからさらに個々の商品I個から確率的に選ばれる。
・Kはクラスタリング数で、自分で決定する
・ここでは顧客数Jと商品数Iは決定している
・顧客ごとに商品グループを選ぶ割合Θは異なる　→　この割合が顧客ごとの特徴となる
・ただし個々の商品の割合は共通

■モデル式
J:顧客人数(定数)
I:商品数(定数)
K:カテゴリ数、自分で決める(定数)
Θ:顧客ごとのカテゴリの割合(確率変数)
φ:カテゴリごとの商品割合、この値の上位割合群の商品が特徴づけているといえる(確率変数)
d:選ばれたカテゴリでone-of-kで表現される(確率変数)
y:I個中、購入した商品でone-of-kで表現される(確率変数)
α:商品yを選ぶ確率分布の事前分布のパラメータ
k ~ Categorical(Θ[j]) → j番目の顧客がカテゴリkを選ぶ
y ~ Categorical(φ[k]) → k番目のカテゴリが選ばれたとき、商品yを選ぶ
φ[k] ~ dirichlet(α) → φの事前分布(カテゴリ分布の共役事前分布だから計算も楽？)
※Θには無情報事前分布をセッティング、Stan標準だと分散がでかめの正規分布

■モデル式の別の表現
以下の表現にしたほうがすっきりする。
j番目の顧客がi番目の商品を購入する確率を以下のようにあらわせる(vはベクトルを表すとする)
P(y=i|顧客=j) = Σ_k:1~K{ Cat(k|Θv[j]) * Cat(i|φv[k]) }
              = Σ_k:1~K{ Θ[j, k] * φ[k, i] } → 結局カテゴリ分布の性質よりこのように簡略化される。
※例えばΘ[j, k]はj番目の顧客がグループkを選ぶ確率
※カテゴリkを選択→商品yを選択 という流れなので、全カテゴリにおける商品yが選ばれる確率を出す必要があるため和となる
※Cat(k|Θ)はk番目の商品が選ばれる確率が出力される確率を表す

■Stan上のモデリング
φ[k] ~ dirichlet(α) → 事前分布
y[i] ~ DoubleCategory(K, Θ[j], φ) → j番目の顧客がi番目の商品を選ぶ確率

■Stanの基礎的な文法
・vector[N] v;
  1つの変数vはN次元ベクトル。
・simplex[N] v;
  非負で合計が1となるN次元ベクトル。
・simplex[N] v[N];
  非負で合計が1となるN次元ベクトルがN個ある。
・*_lpmf(y, ...)
  独自の確率質量の定義。確率密度の場合*_lpdfとなる。
  一番目の引数は独自確率密度に従う変数。
  例として正規分布であればnormal_lpdf(y, μ, σ^2)となる。
  戻り値として、対数分布を返す必要がある。
  これを呼び出す場合は y ~ normal(μ, σ^2)といった具合になる。
  つまり、y ~ normal(μ, σ^2)　と　target += normal_lpdf(y|μ, σ^2)は同じ。
  stanは内部で対数事後確率にして偏微分しやすくしてる。
・log_sum_exp()
   log_sum_exp(Xv) = log{ exp(X1) + exp(X2) + ... + exp(Xn) }
   和の状態になっているものに対してlogを取りたいときに使う。
   例えば、log{ ΣXv } を計算したいとする。
   各Xiをlog変換して合計値算出し、log_sum_expにその合計値を入れれば実現できる。
   例として、確率分布が{ X1*Y1 + X2*Y2 + ... } とかだと偏微分しづらいので、
   この場合 Z1 = log(X1) + log(Y1)に変換して全Zvを足し算した後、log_sum_exp(Zv)とすれば、
   log { ΣZv }が求められる。
"""


def main():
    df = pd.read_csv('./data/data_lda.csv')
    person_id = df['PersonID'].tolist()  # 一意の顧客ID
    item_id = df['ItemID'].tolist()  # 一意の商品ID
    N = len(df)  # 全イベント数
    K = 6  # カテゴリ数
    J = max(person_id)  # 全顧客数
    I = max(item_id)  # 全商品数
    alpha_for_theta = [0.8] * K  # Θの事前分布のパラメータ → これまで予測すると収束しない
    alpha_for_phi = [0.2] * I  # φの事前分布のパラメータ → こいつまで予測すると収束しない

    data = {
        'N': N,
        'K': K,
        'J': J,
        'I': I,
        'person_id': person_id,
        'item_id': item_id,
        'alpha_for_theta': alpha_for_theta,
        'alpha_for_phi': alpha_for_phi
    }

    code = """
    functions {
        real DoubleCategorical_lpmf(int i, int K, int j, vector[] theta, vector[] phi) {
            vector[K] log_prob_vec;
            for (k in 1:K)
                log_prob_vec[k] = log(theta[j, k]) + log(phi[k, i]);
            return log_sum_exp(log_prob_vec);
        }
    }

    data {
        int<lower=1> N;
        int<lower=1> K;
        int<lower=1> J;
        int<lower=1> I;
        int<lower=1, upper=J> person_id[N];
        int<lower=1, upper=I> item_id[N];
        vector<lower=0>[I] alpha_for_phi;
    }

    parameters {
        simplex[K] theta[J];
        simplex[I] phi[K];        
    }

    model {
        for (k in 1:K)
            phi[k] ~ dirichlet(alpha_for_phi);
        for (n in 1:N)
            item_id[n] ~ DoubleCategorical(K, person_id[n], theta, phi);
    }
    """

    model = pystan.StanModel(model_code=code)
    fit = model.sampling(data=data, iter=1000, chains=2)
    la = fit.extract(permuted=True)
    print(fit)


if __name__ == '__main__':
    main()

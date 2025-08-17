# %% [markdown]
# 

# %% [markdown]
# #### 本Notebookは書籍『Pythonで動かして学ぶ！Kaggleデータ分析入門』(翔泳社, 2020)の内容のサンプルコードとなります。

# %% [markdown]
# ## 3.4　データ分析の準備をする

# %% [markdown]
# ### ［手順4］ライブラリをインストール・インポートする

# %% [markdown]
# #### Anaconda（Windows）、macOSの場合

# %% [markdown]
# リスト3.1　ライブラリのインポート

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ### ［手順5］データを読み込む

# %% [markdown]
# #### Anaconda（Windows）、macOSの場合

# %% [markdown]
# リスト3.2　データの読み込み

# %%
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
submission = pd.read_csv("./data/gender_submission.csv")

# %% [markdown]
# リスト3.3　train.csvのデータの概要を表示

# %%
train_df

# %% [markdown]
# リスト3.4　test.csvのデータの概要を表示

# %%
test_df

# %% [markdown]
# リスト3.5　gender_submission.csvのデータの概要を表示

# %%
submission

# %% [markdown]
# ### ［手順6］ランダムシードを設定する

# %% [markdown]
# リスト3.7　ランダムシードの設定

# %%
import random
np.random.seed(1234)
random.seed(1234)

# %% [markdown]
# ## 3.4　データの概要を把握する

# %% [markdown]
# #### データ数を確認する

# %% [markdown]
# リスト3.8　行数、列数の表示

# %%
print(train_df.shape)
print(test_df.shape)

# %% [markdown]
# #### データの先頭行を確認する

# %% [markdown]
# リスト3.9　データの中身の確認

# %%
pd.set_option("display.max_columns",50)
pd.set_option("display.max_rows",50)

# %%
train_df.head()

# %%
test_df.head()

# %% [markdown]
# #### データの型を確認する

# %% [markdown]
# リスト3.10　データ内の各列の値の型を参照

# %%
train_df.dtypes

# %% [markdown]
# #### データの統計量を確認する

# %% [markdown]
# リスト3.11　train.csvの数値データの概要を確認

# %%
train_df.describe()

# %% [markdown]
# リスト3.12　test.csvの数値データの概要を確認

# %%
test_df.describe()

# %% [markdown]
# #### カテゴリ変数を確認する

# %% [markdown]
# リスト3.13　各カテゴリ変数の確認

# %%
train_df["Sex"].value_counts()

# %%
train_df["Embarked"].value_counts()

# %%
train_df["Cabin"].value_counts()

# %% [markdown]
# #### 欠損値を確認する

# %% [markdown]
# リスト3.14　各変数の欠損値の確認

# %%
train_df.isnull().sum()

# %%
test_df.isnull().sum()

# %% [markdown]
# ## 3.6　データを可視化する

# %% [markdown]
# #### 可視化用のライブラリをインストール・インポートする

# %% [markdown]
# リスト3.15　データを可視化するライブラリのインポート

# %%
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# #### 表示結果の書式を指定する

# %% [markdown]
# リスト3.16　ggplotを指定する

# %%
plt.style.use("ggplot")

# %% [markdown]
# ### Survivedに関するデータを可視化する

# %% [markdown]
# #### DataFrameから任意の列を抽出する

# %% [markdown]
# リスト3.17　DataFrameからEmbarked、Survived、PassengerIdの列を抽出

# %%
train_df[["Embarked","Survived","PassengerId"]]

# %% [markdown]
# #### 可視化したいデータから欠損値を除外する

# %% [markdown]
# リスト3.18　可視化したいデータから欠損値を除外

# %%
train_df[["Embarked","Survived","PassengerId"]].dropna()

# %% [markdown]
# #### EmbarkedとSurvivedの値で集計する

# %% [markdown]
# リスト3.19　EmbarkedとSurvivedの値で集計

# %%
train_df[["Embarked","Survived","PassengerId"]].dropna().groupby(["Embarked","Survived"]).count()

# %% [markdown]
# #### データを横持ちに変換する

# %% [markdown]
# リスト3.20　データを横持ちに変換

# %%
embarked_df = train_df[["Embarked","Survived","PassengerId"]].dropna().groupby(["Embarked","Survived"]).count().unstack()

# %%
embarked_df

# %% [markdown]
# #### 積み上げ縦棒グラフで可視化する

# %% [markdown]
# リスト3.21　積み上げ縦棒グラフで可視化

# %%
embarked_df.plot.bar(stacked=True)

# %% [markdown]
# #### 数値で確認する

# %% [markdown]
# リスト3.22　新たにsurvived_rateという変数で数値を確認

# %%
embarked_df["survived_rate"]=embarked_df.iloc[:,0]/(embarked_df.iloc[:,0] + embarked_df.iloc[:,1])

# %%
embarked_df

# %% [markdown]
# #### 性別やチケットの階級について可視化する

# %% [markdown]
# リスト3.23　性別やチケットの階級を可視化

# %%
sex_df = train_df[["Sex","Survived","PassengerId"]].dropna().groupby(["Sex","Survived"]).count().unstack()
sex_df.plot.bar(stacked=True)

# %%
ticket_df = train_df[["Pclass","Survived","PassengerId"]].dropna().groupby(["Pclass","Survived"]).count().unstack()
ticket_df.plot.bar(stacked=True)

# %% [markdown]
# #### 年代ごとの生存率をヒストグラムで可視化する

# %% [markdown]
# リスト3.24　年代ごとの生存率をヒストグラムで可視化

# %%
# plt.hist((train_df[train_df["Survived"] == 0][["Age"]].values, train_df[train_df["Survived"] == 1][["Age"]].values),
# histtype="barstacked", bins=8, label=("Death", "Survive"))
# plt.legend()

age_dead      = train_df.loc[train_df["Survived"] == 0, "Age"].dropna()
age_survived  = train_df.loc[train_df["Survived"] == 1, "Age"].dropna()

plt.hist(
    [age_dead, age_survived], 
    histtype="barstacked",
    bins=8,
    label=("Death", "Survived")
    )

plt.legend()

# %% [markdown]
# #### カテゴリ変数をダミー変数化する

# %% [markdown]
# リスト3.25　カテゴリ変数をダミー変数化

# %%
train_df_corr = pd.get_dummies(train_df, columns=["Sex"],drop_first=True)
train_df_corr = pd.get_dummies(train_df_corr, columns=["Embarked"])

# %%
train_df_corr.head()

# %% [markdown]
# #### 相関行列を作成する

# %% [markdown]
# リスト3.26　相関行列の作成

# %%
# train_corr = train_df_corr.corr()

train_df_corr_selected = train_df_corr.select_dtypes(include=["number"])
train_corr = train_df_corr_selected.corr()

train_corr

# %%
train_corr

# %% [markdown]
# #### ヒートマップで可視化する

# %% [markdown]
# リスト3.27　ヒートマップで可視化

# %%
plt.figure(figsize=(9, 9))
sns.heatmap(train_corr, vmax=1, vmin=-1, center=0, annot=True)

# %% [markdown]
# ## 3.7 前処理・特徴量の生成を行う

# %% [markdown]
# #### 学習データとテストデータを統合する

# %% [markdown]
# リスト3.28　学習データとテストデータを統合したものを作成

# %%
all_df = pd.concat([train_df, test_df],sort=False).reset_index(drop=True)

# %%
all_df

# %% [markdown]
# ### 全体データで欠損値の数を確認する

# %% [markdown]
# リスト3.29　全体データで欠損値の数を確認

# %%
all_df.isnull().sum()

# %% [markdown]
# #### 欠損データを穴埋めする（Fare）

# %% [markdown]
# リスト3.30　PclassごとのFareの平均値を計算

# %%
Fare_mean = all_df[["Pclass","Fare"]].groupby("Pclass").mean().reset_index()

# %% [markdown]
# リスト3.31　カラム名の変更

# %%
Fare_mean.columns = ["Pclass","Fare_mean"]

# %%
Fare_mean

# %% [markdown]
# リスト3.32　欠損値を置き換える

# %%
all_df = pd.merge(all_df, Fare_mean, on="Pclass",how="left")
all_df.loc[(all_df["Fare"].isnull()), "Fare"] = all_df["Fare_mean"]
all_df = all_df.drop("Fare_mean",axis=1)

# %% [markdown]
# #### Nameの敬称に注目する

# %% [markdown]
# リスト3.33　Nameの欠損値を調べる

# %%
all_df["Name"].head(5)

# %% [markdown]
# #### 敬称を変数として追加する

# %% [markdown]
# リスト3.34　敬称を変数として追加

# %%
# name_df = all_df["Name"].str.split("[,.]",2,expand=True)

name_df = all_df["Name"].str.split(pat="[,.]", n=2, expand=True)

# %% [markdown]
# リスト3.35　カラム名の変更

# %%
name_df.columns = ["family_name","honorific","name"]

# %%
name_df

# %% [markdown]
# リスト3.36　先頭と末尾の空白文字の削除

# %%
name_df["family_name"] =name_df["family_name"].str.strip()
name_df["honorific"] =name_df["honorific"].str.strip()
name_df["name"] =name_df["name"].str.strip()

# %% [markdown]
# #### 各敬称ごとの人数をカウントする

# %% [markdown]
# リスト3.37　各honorific（敬称）ごとの人数をカウント

# %%
name_df["honorific"].value_counts()

# %% [markdown]
# #### 敬称ごとの年齢分布を確認する

# %% [markdown]
# リスト3.38　2つのDataFrameを横に結合

# %%
all_df = pd.concat([all_df, name_df],axis=1)

# %%
all_df

# %% [markdown]
# リスト3.39　敬称ごとの年齢の分布を確認

# %%
plt.figure(figsize=(18, 5))
sns.boxplot(x="honorific", y="Age", data=all_df)

# %% [markdown]
# #### 敬称ごとの年齢の平均値を確認する

# %% [markdown]
# リスト3.40　敬称ごとの年齢の平均値を確認

# %%
all_df[["Age","honorific"]].groupby("honorific").mean()

# %% [markdown]
# #### 敬称ごとの生存率の違いについて確認する

# %% [markdown]
# リスト3.41　もとのDataFrameに名前のDataFrameを結合

# %%
train_df = pd.concat([train_df,name_df[0:len(train_df)].reset_index(drop=True)],axis=1)
test_df = pd.concat([test_df,name_df[len(train_df):].reset_index(drop=True)],axis=1)

# %% [markdown]
# リスト3.42　敬称ごとにSurvivedの値ごとの人数を集計

# %%
honorific_df = train_df[["honorific","Survived","PassengerId"]].dropna().groupby(["honorific","Survived"]).count().unstack()
honorific_df.plot.bar(stacked=True)

# %% [markdown]
# #### 年齢が欠損しているものは、敬称ごとの平均年齢で補完する

# %% [markdown]
# リスト3.43　敬称ごとの平均年齢で年齢が欠損しているデータを穴埋めする

# %%
honorific_age_mean = all_df[["honorific","Age"]].groupby("honorific").mean().reset_index()
honorific_age_mean.columns = ["honorific","honorific_Age"]

# %%
all_df = pd.merge(all_df, honorific_age_mean, on="honorific", how="left")
all_df.loc[(all_df["Age"].isnull()), "Age"] = all_df["honorific_Age"]
all_df = all_df.drop(["honorific_Age"],axis=1)

# %% [markdown]
# #### 家族人数を追加する

# %% [markdown]
# リスト3.44　家族に関する変数を足して家族人数とする

# %%
all_df["family_num"] = all_df["Parch"] + all_df["SibSp"]

# %%
all_df["family_num"].value_counts()

# %% [markdown]
# #### 同船している家族人数が0人（1人乗船）かどうかを表すaloneという変数を追加する

# %% [markdown]
# リスト3.45　1人か同船家族がいるかを変数に加える

# %%
all_df.loc[all_df["family_num"] ==0, "alone"] = 1
all_df["alone"].fillna(0, inplace=True)

# %% [markdown]
# #### 不要な変数を削除する

# %% [markdown]
# リスト3.46　不要な変数の削除

# %%
all_df = all_df.drop(["PassengerId","Name","family_name","name","Ticket","Cabin"],axis=1)

# %%
all_df.head()

# %% [markdown]
# #### カテゴリ変数を数値に変換する

# %% [markdown]
# リスト3.47　変数の型がobjectであるものをカテゴリ変数として管理

# %%
categories = all_df.columns[all_df.dtypes == "object"]
print(categories)

# %% [markdown]
# #### 敬称はMr、Miss、Mrs、Master以外は数が少ないため、 otherとして統合する

# %% [markdown]
# リスト3.48　敬称はMr、Miss、Mrs、Master以外は数が少ないため、otherとして統合

# %%
all_df.loc[~((all_df["honorific"] =="Mr") |
    (all_df["honorific"] =="Miss") |
    (all_df["honorific"] =="Mrs") |
    (all_df["honorific"] =="Master")), "honorific"] = "other"

# %%
all_df.honorific.value_counts()

# %% [markdown]
# #### 機械学習用のライブラリをインストール・インポートする

# %% [markdown]
# リスト3.49　LabelEncoderのインポート

# %%
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# リスト3.50　LabelEncodingの実行①

# %%
all_df["Embarked"].fillna("missing", inplace=True)

# %%
all_df.head()

# %%
le = LabelEncoder()
le = le.fit(all_df["Sex"])
all_df["Sex"] = le.transform(all_df["Sex"])

# %% [markdown]
# リスト3.51　LabelEncodingの実行②

# %%
for cat in categories:
    le = LabelEncoder()
    print(cat)
    if all_df[cat].dtypes == "object":    
        le = le.fit(all_df[cat])
        all_df[cat] = le.transform(all_df[cat])

# %%
all_df.head()

# %% [markdown]
# #### すべてのデータを学習データとテストデータに戻す

# %% [markdown]
# リスト3.52　train/testデータセットにデータを戻す

# %%
train_X = all_df[~all_df["Survived"].isnull()].drop("Survived",axis=1).reset_index(drop=True)
train_Y = train_df["Survived"]
test_X = all_df[all_df["Survived"].isnull()].drop("Survived",axis=1).reset_index(drop=True)

# %% [markdown]
# ## 3.8 モデリングを行う

# %% [markdown]
# #### LightGBMのライブラリをインストール・インポートする

# %% [markdown]
# リスト3.53　LightGBMのライブラリのインポート

# %%
import lightgbm as lgb

# %% [markdown]
# #### ホールドアウト、クロスバリデーションを行うための ライブラリをインポートする

# %% [markdown]
# リスト3.54　ホールドアウト、クロスバリデーションを行うためのライブラリをインポート

# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# %% [markdown]
# #### 学習データの20%を検証データに分割する

# %% [markdown]
# リスト3.55　学習データの20%を検証データに分割する

# %%
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.2)

# %% [markdown]
# #### LightGBM用のデータセットを作成する

# %% [markdown]
# リスト3.56　カテゴリ変数を指定してLightGBM用のデータセットを作成

# %%
categories = ["Embarked", "Pclass", "Sex","honorific","alone"]

# %%
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categories)
lgb_eval = lgb.Dataset(X_valid, y_valid,  categorical_feature=categories, reference=lgb_train)

# %% [markdown]
# #### ハイパーパラメータを設定する

# %% [markdown]
# リスト3.57　ハイパーパラメータの設定

# %%
lgbm_params = {
    "objective":"binary",        
    "random_seed":1234
}

# %% [markdown]
# #### LightGBMによる機械学習モデルを学習させる

# %% [markdown]
# リスト3.58　機械学習モデルの学習

# %%
# model_lgb = lgb.train(lgbm_params, 
#                       lgb_train, 
#                       valid_sets=lgb_eval, 
#                       num_boost_round=100,
#                       early_stopping_rounds=20,
#                       verbose_eval=10)

model_lgb = lgb.train(
    lgbm_params,
    lgb_train,
    valid_sets=lgb_eval,
    num_boost_round=100,
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(period=10)]
    )

# %% [markdown]
# #### 各変数の重要度を調べる

# %% [markdown]
# リスト3.59　各変数の重要度の確認

# %%
model_lgb.feature_importance()

# %% [markdown]
# リスト3.60　もとのデータのカラム名を表示

# %%
importance = pd.DataFrame(model_lgb.feature_importance(), index=X_train.columns, columns=["importance"]).sort_values(by="importance",ascending =True)
importance.plot.barh()

# %% [markdown]
# ### 検証データで予測精度を確認

# %% [markdown]
# #### モデルを検証データに適用する

# %% [markdown]
# リスト3.61　モデルを検証データに適用

# %%
y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

# %% [markdown]
# #### accuracyを計算するライブラリをインポートする

# %% [markdown]
# リスト3.62　accuracyを計算するライブラリのインポート

# %%
from sklearn.metrics import accuracy_score

# %%
accuracy_score(y_valid, np.round(y_pred))

# %% [markdown]
# #### ハイパーパラメータを変更する

# %% [markdown]
# リスト3.63　ハイパーパラメータの値の変更

# %%
lgbm_params = {
    "objective":"binary",
    "max_bin":331,
    "num_leaves": 20,
    "min_data_in_leaf": 57,
    "random_seed":1234
}

# %% [markdown]
# リスト3.64　再度LightGBMのデータセットを指定し、学習を実行

# %%
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categories)
lgb_eval = lgb.Dataset(X_valid, y_valid, categorical_feature=categories, reference=lgb_train)

# %%
model_lgb = lgb.train(lgbm_params, lgb_train, 
                      valid_sets=lgb_eval, 
                      num_boost_round=100,
                      early_stopping_rounds=20,
                      verbose_eval=10)

# %% [markdown]
# リスト3.65　検証データに対する予測値を算出

# %%
y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)

# %% [markdown]
# リスト3.66　精度の計算

# %%
accuracy_score(y_valid, np.round(y_pred))

# %% [markdown]
# ### クロスバリデーションによる学習

# %% [markdown]
# リスト3.67　3分割（3-fold）する

# %%
folds = 3

kf = KFold(n_splits=folds)

# %% [markdown]
# リスト3.68　クロスバリデーションによる学習

# %%
models = []

for train_index, val_index in kf.split(train_X):
    X_train = train_X.iloc[train_index]
    X_valid = train_X.iloc[val_index]
    y_train = train_Y.iloc[train_index]
    y_valid = train_Y.iloc[val_index]
        
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categories)
    lgb_eval = lgb.Dataset(X_valid, y_valid, categorical_feature=categories, reference=lgb_train)    
    
    model_lgb = lgb.train(lgbm_params, 
                          lgb_train, 
                          valid_sets=lgb_eval, 
                          num_boost_round=100,
                          early_stopping_rounds=20,
                          verbose_eval=10,
                         )
    
    
    y_pred = model_lgb.predict(X_valid, num_iteration=model_lgb.best_iteration)
    print(accuracy_score(y_valid, np.round(y_pred)))
    
    models.append(model_lgb)    

# %% [markdown]
# #### テストデータにおける予測結果を算出する

# %% [markdown]
# リスト3.69　テストデータの結果を予測して格納

# %%
preds = []

for model in models:
    pred = model.predict(test_X)
    preds.append(pred)

# %% [markdown]
# #### 予測結果の平均をとる

# %% [markdown]
# リスト3.70　予測結果の平均をとる

# %%
preds_array = np.array(preds)
preds_mean = np.mean(preds_array, axis=0)

# %% [markdown]
# リスト3.71　0か1に変換

# %%
preds_int = (preds_mean > 0.5).astype(int)

# %% [markdown]
# #### submissionファイルを生成する

# %% [markdown]
# リスト3.72　submissionファイルの「Survived」の値を置き換え

# %%
submission["Survived"] = preds_int

# %%
submission

# %% [markdown]
# #### 結果をCSVとして書き出す（Anaconda(Windows）、macOSの場合）

# %% [markdown]
# リスト3.73　CSVファイルとして書き出す

# %%
submission.to_csv("./submit/titanic_submit01.csv",index=False)

# %% [markdown]
# ## 3.10　精度以外の分析視点

# %% [markdown]
# ### 追加分析❶：Titanicにはどのような人が乗船していたのか

# %% [markdown]
# #### チケットクラスごとの人数を確認する

# %% [markdown]
# リスト3.75　データの読み込み（Anaconda（Windows）やmacOSのJupyterNotebookの場合）

# %%
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
all_df = pd.concat([train_df, test_df],sort=False).reset_index(drop=True)

# %% [markdown]
# リスト3.77　チケットクラスごとの人数の確認

# %%
all_df.Pclass.value_counts()

# %% [markdown]
# リスト3.78　リスト3.77の結果を可視化

# %%
all_df.Pclass.value_counts().plot.bar()

# %% [markdown]
# #### 料金の分布を確認する

# %% [markdown]
# リスト3.79　チケットクラスごとの料金の分布を確認

# %%
all_df[["Pclass","Fare"]].groupby("Pclass").describe()

# %% [markdown]
# リスト3.80　チケットクラスごとの料金の分布を可視化

# %%
plt.figure(figsize=(6, 5))
sns.boxplot(x="Pclass", y="Fare", data=all_df)

# %% [markdown]
# #### 1等級チケットのうち、高額チケット(1等級チケットの上位25%)をPclass0にする

# %% [markdown]
# リスト3.81　Pclass2という変数の作成

# %%
all_df["Pclass2"] = all_df["Pclass"]

# %% [markdown]
# #### class2のうちFareが108より大きいものを0に変更する

# %% [markdown]
# リスト3.82　Fareが108より大きいものを0に変更

# %%
all_df.loc[all_df["Fare"]>108, "Pclass2"] = 0

# %%
all_df[all_df["Pclass2"] == 0]

# %% [markdown]
# #### チケットクラスごとの年齢の分布を確認する

# %% [markdown]
# リスト3.83　年齢の分布を確認

# %%
all_df[["Pclass2","Age"]].groupby("Pclass2").describe()

# %%
plt.figure(figsize=(6, 5))
sns.boxplot(x="Pclass2", y="Age", data=all_df)

# %% [markdown]
# #### 15歳より上の人に限定して再度確認する

# %% [markdown]
# リスト3.84　15歳より上に限った年齢の分布を確認

# %%
all_df[all_df["Age"]>15][["Pclass2","Age"]].groupby("Pclass2").describe()

# %%
plt.figure(figsize=(6, 5))
sns.boxplot(x="Pclass2", y="Age", data=all_df[all_df["Age"]>15])

# %% [markdown]
# #### 年齢と乗船料金の分布を確認する

# %% [markdown]
# リスト3.85　年齢と乗船料金の分布

# %%
all_df.plot.scatter(x="Age", y="Fare", alpha=0.5)

# %% [markdown]
# #### チケットクラスごとの乗船家族人数を確認する

# %% [markdown]
# リスト3.86　チケットクラスによって乗船家族人数に違いがあるか確認

# %%
all_df["family_num"] = all_df["SibSp"] + all_df["Parch"]

# %%
all_df[["Pclass2","family_num"]].groupby("Pclass2").describe()

# %%
plt.figure(figsize=(6, 5))
sns.boxplot(x="Pclass2", y="family_num", data=all_df)

# %% [markdown]
# #### チケットクラスごとの男女比について確認する

# %% [markdown]
# リスト3.87　男女比について確認

# %%
Pclass_gender_df = all_df[["Pclass2","Sex","PassengerId"]].dropna().groupby(["Pclass2","Sex"]).count().unstack()

# %%
Pclass_gender_df.plot.bar(stacked=True)

# %%
Pclass_gender_df["male_ratio"] = Pclass_gender_df["PassengerId", "male"] / (Pclass_gender_df["PassengerId", "male"] + Pclass_gender_df["PassengerId", "female"])

# %%
Pclass_gender_df

# %% [markdown]
# #### 港ごとの違いを確認

# %% [markdown]
# リスト3.88　港ごとの違いを確認

# %%
Pclass_emb_df = all_df[["Pclass2","Embarked","PassengerId"]].dropna().groupby(["Pclass2","Embarked"]).count().unstack()

# %%
Pclass_emb_df = Pclass_emb_df.fillna(0)

# %%
Pclass_emb_df.plot.bar(stacked=True)

# %% [markdown]
# リスト3.89　100%積み上げ縦棒グラフに変換

# %%
Pclass_emb_df_ratio = Pclass_emb_df.copy()
Pclass_emb_df_ratio["sum"] = Pclass_emb_df_ratio["PassengerId","C"] + Pclass_emb_df_ratio["PassengerId","Q"] + Pclass_emb_df_ratio["PassengerId","S"]
Pclass_emb_df_ratio["PassengerId","C"] = Pclass_emb_df_ratio["PassengerId","C"] / Pclass_emb_df_ratio["sum"]
Pclass_emb_df_ratio["PassengerId","Q"] = Pclass_emb_df_ratio["PassengerId","Q"] / Pclass_emb_df_ratio["sum"]
Pclass_emb_df_ratio["PassengerId","S"] = Pclass_emb_df_ratio["PassengerId","S"] / Pclass_emb_df_ratio["sum"]
Pclass_emb_df_ratio = Pclass_emb_df_ratio.drop(["sum"],axis=1)

# %%
Pclass_emb_df_ratio

# %%
Pclass_emb_df_ratio.plot.bar(stacked=True)

# %% [markdown]
# ### 追加分析❷：特定のクラスタに注目してみる

# %% [markdown]
# リスト3.90　「Cherbourgからの1人乗船の若者」というクラスタの特徴を分析

# %%
C_young10 = all_df[(all_df["Embarked"] == "C") & (all_df["Age"] // 10 == 1) & (all_df["family_num"] == 0)]

# %%
C_young20 = all_df[(all_df["Embarked"] == "C") & (all_df["Age"] // 10 == 2) & (all_df["family_num"] == 0)]

# %%
len(C_young10)

# %%
len(C_young20)

# %% [markdown]
# #### Cherbourgの若者の乗船料金の分布を調べる

# %% [markdown]
# リスト3.91　全体の中における「Cherbourgからの1人乗船の若者（10代）」を確認

# %%
ax = all_df.plot.scatter(x="Age", y="Fare", alpha=0.5)
C_young10.plot.scatter(x="Age", y="Fare", color="red",alpha=0.5, ax=ax)

# %% [markdown]
# リスト3.92　1人乗船の人に限った中での「Cherbourgからの1人乗船の若者（10代）」を確認

# %%
ax = all_df[all_df["family_num"] == 0].plot.scatter(x="Age", y="Fare", alpha=0.5)
C_young10.plot.scatter(x="Age", y="Fare", color="red",alpha=0.5, ax=ax)

# %% [markdown]
# リスト3.93　「Cherbourgからの1人乗船した20代」についても同様に確認

# %%
ax = all_df.plot.scatter(x="Age", y="Fare", alpha=0.5)
C_young20.plot.scatter(x="Age", y="Fare", color="red",alpha=0.5, ax=ax)

# %%
ax = all_df[all_df["family_num"] == 0].plot.scatter(x="Age", y="Fare", alpha=0.5)
C_young20.plot.scatter(x="Age", y="Fare", color="red",alpha=0.5, ax=ax)

# %% [markdown]
# リスト3.94　Cherbourgからの乗船客を全体の中で表示する

# %%
C_all = all_df[(all_df["Embarked"] == "C")]
ax = all_df.plot.scatter(x="Age", y="Fare", alpha=0.5)
C_all.plot.scatter(x="Age", y="Fare", color="red",alpha=0.5, ax=ax)

# %% [markdown]
# #### 各乗船港ごとに10代1人乗船客の平均料金を比較する

# %% [markdown]
# リスト3.95　各乗船港ごとに10代1人乗船客の平均料金を比較

# %%
all_df[(all_df["Age"] // 10 == 1) & (all_df["family_num"]== 0)][["Embarked","Fare"]].groupby("Embarked").mean()



---
title: basics
---
## pandasでできること

### データの入出力

CSVやExcel、DBなど、さまざまな形式のデータソースから表形式のデータを読み込めます。また、pandasで処理したデータを、各形式のファイルに書き込めます。

### データの絞り込み

「列Aが 100 以上の行だけを抽出する」など、指定した条件に合致するデータだけを絞り込めます。

### データのクリーニング

「身長のデータなのに負の値になっている」といった異常な値をチェック・修正したり、欠損値を補間・削除するなど、「おかしなデータ」を分析可能な形に整備できます。

### データの集約

「店舗ごとの売り上げの合計値を計算する」「ユーザの年齢ごとに契約日数の平均値を計算する」など、グループごとに特定の処理を適用できます。

### データの統合

複数のデータをひとつのデータに統合します。単純な連結のほか、SQLのJOINのような異なる種類のデータも結合できます。

### データの分析・可視化

各列の基本統計量を計算したり、棒グラフや散布図などいろいろなグラフを作成できます。

## pandasの基本的なデータ構造

- pandasの基本のデータ構造には、DataFrameとSeriesがあります。
- DataFrameは、2次元の表形式のデータです。pandas.DataFrameクラスを使います。
- Seriesは1次元のデータで、表の中の1列のようなイメージです。pandas.Seriesクラスを使います。

``` python
# DataFrameの作成
df = pd.DataFrame(
    [["佐藤", 172, 60], ["田中", 160, 50], ["鈴木", 165, 58]],  # リストのリスト
    columns=["Name", "Height", "Weight"],  # 列名のリスト
)

df.columns

type(df["Name"])
```

## データの読み込み

pandasには、いろいろなデータソースに対応した入出力機能が備わっています。たとえば、CSVデータを読み込む時は read_csv() を使います。

### データソース

- CSVファイル（.csv）
- 区切りのあるテキストファイル
- HTML内のtableタグ（.html）
- JSON（.json）
- リレーショナルデータベース
- Google BigQuery

``` python
import pandas as pd

# CSVファイルからデータを読み込み
df = pd.read_csv("dataset/physical_measurement.csv")
```

## DataFrameの確認

- DataFrameの先頭数行を表示（head()）
- DataFrameの末尾数行を表示（tail()）
- DataFrameの形状（行数と列数）（shape）
- 列名一覧 （columns）
- 各列の型　（dtypes）
- 基本統計量 （describe()）

``` python
import pandas as pd

# データの読み込み
df = pd.read_csv("dataset/physical_measurement.csv")

df.head()
df.tail()
df.dtypes
df.shape
df.columns
df.describe()
```

## データの絞り込み

pandasのDataFrameでは、df[条件式] のように書くことで条件に一致する行を絞り込めます。

たとえばdf[df["A"] >= 30]とすると、「列Aが30以上の行」だけを絞り込みます。条件式にはPythonの比較同様、==、!=、>、<、>=、<=などが使えます。

``` python
import pandas as pd

# データの読み込み
df = pd.read_csv("dataset/physical_measurement.csv")

# 身長が170cm以上の行だけを抽出
df_filtering = df[df["Height"] >= 170]
```

## データのクリーニングとは

### おかしなデータ

- 体重を表すデータなのに、負の値が入っている（範囲外の値）
- システムの障害により、特定の期間のデータだけが記録されていない（データの欠損）
- アンケートの仕様が変わったことにより、変更前と変更後で回答項目が表記揺れしている（表記揺れ）
- データ仕様上ありえないはずなのに、まったく同じ値の行が重複して存在する（データの重複）

### チェックの例

- 詳細なデータ分析に入る前に「おかしなデータ」がないかチェックし、対処することが重要です。
- たとえば、データの最大値・最小値をチェックすると、想定しない範囲のデータに気付きやすくなります。
  - おかしなデータへの対処方法には、「データごと削除する」「データの値を置換する」などがあります。
- このようなデータを整備する工程のことを、データのクリーニングと呼びます。

- データの最大値・最小値を確認し、期待する範囲外の値がないか確認する
- 欠損値の有無を確認する
- データの種類を確認し、意図しない値や表記揺れがないか確認する
- データの分布を確認し、不自然な点がないか確認する
- データの重複を確認し、本来なら発生しないような不自然な重複がないか確認する

``` python
import pandas as pd

# データの読み込み
df = pd.read_csv("dataset/physical_measurement_dirty.csv")

df.describe()

# 身長が0以下の行だけを選択
df[df["Height"] <= 0]

# 列Heightが0より大きい行だけを抽出
df_after = df[df["Height"] > 0]

print(df.shape)  # 削除前のDataFrameの行数・列数
print(df_after.shape)  # 削除後のDataFrameの行数・列数

print(df["Height"].min())  # 削除前の身長の最小値
print(df_after["Height"].min())  # 削除後の身長の最小値
```

## 欠損値の除去

### データが存在しないパターン

- 入力者のミスでデータが入力されていない
- システムのエラーによりデータが記録されなかった
- アンケートで未回答だった

pandasには、欠損値の確認や対応のための機能がいろいろ備わっています。たとえば、次のような関数です。

- 欠損値を確認する: info(), isna()など
- 欠損値を含む行を削除する: dropna()
- 欠損値を他の値で置換する: fillna()

``` python
import pandas as pd

# データの読み込み
df = pd.read_csv("dataset/physical_measurement_dirty.csv")

df.shape
df.info()
df["Weight"].isna()
# 列Weightに欠損値を含む行を絞り込んで表示
df[df["Weight"].isna()]
# 列Weightに欠損値がある行を削除する
df_after = df.dropna(subset=["Weight"])
print(df.shape)  # 削除前のDataFrameの行数・列数
print(df_after.shape)  # 削除後のDataFrameの行数・列数
df_after[df_after["Weight"].isna()]
```

## データの置換

- データクリーニングでおかしなデータが見つかったとき、適切な値がわかっていればデータを置換して対処することもあります。
- pandasには、特定の条件のデータを置換するための機能がいろいろ備わっています。
  - たとえばdf[列名].mask(条件式, 置換後の値)とすると、指定した条件に一致する行の値を置換後の値で置き換えます。
- df[既存の列] = 更新後の値 のように書くことで、既存の列を更新できます。

pandasで特定のデータを置換するには、たとえば次のようなメソッドがあります。

mask(条件式, 置換後の値): 指定した条件に一致するデータを置換する
where(条件式, 置換後の値): 指定した条件に一致しないデータを置換する
apply(関数): 全ての行に対して、指定した関数を適用する。より複雑なルールで値を置換したいときに便利。

たとえばmask()を使って「列Aが負の値の場合、列Aの値を0で置換する」ケースでは、`df["A"].mask(df["A"] < 0, 0)`のように書きます。

``` python

# 列Clubの値の種類と件数を表示
df["Club"].value_counts()

# 列Clubの値が「無所属」に一致するものを「所属なし」に置換する。
df["Club"].mask(df["Club"] == "無所属", "所属なし")

# 置換後の列を代入
df["Club"] = df["Club"].mask(df["Club"] == "無所属", "所属なし")

```

## 列の追加と演算

`df[新しい列名] = 新しい列の値`
`df[既存の列名] = 更新後の値` これだと更新になる。

`df["C"] = df["A"] + df["B"]` のように書くと、「列Aと列Bの各データを足し合わせたSeries」が生成されます

``` python
import pandas as pd

# データの読み込み
df = pd.read_csv("dataset/physical_measurement_clean.csv")

# 新しい列Classの作成（全て"1年A組"）
df["Class"] = "1年A組"

# 100で割って身長の単位をcmからmに変換
df["Height"] / 100

# 新しい列HeightMを作成
df["HeightM"] = df["Height"] / 100

# 列の値を使ってBMIを計算
df["Weight"] / df["HeightM"] ** 2

# 新しい列BMIを作成
df["BMI"] = df["Weight"] / df["HeightM"] ** 2
```

## DataFrameの連結

たとえばconcat()を使うと、DataFrameを縦また横方向に連結できます。引数axisで0を指定すると縦方向に、1を指定すると横方向に連結します（デフォルトでは0のため、未指定だと縦方向の連結になります）。

またmerge()を使うと、基準となるキーを指定して複数の種類のデータを結合できます。SQLを知っている方は、JOINのような結合をイメージしてください。

``` python
import pandas as pd
# 1年A組のデータを読み込み
df_a = pd.read_csv("dataset/physical_measurement_A.csv")

# 新しい列Classを追加（全ての行に"1年A組"が格納される）
df_a["Class"] = "1年A組"

# 1年B組のデータを読み込み
df_b = pd.read_csv("dataset/physical_measurement_B.csv")

# 新しい列Classを追加（全ての行に"1年B組"が格納される）
df_b["Class"] = "1年B組"

# df_aとdf_bを連結して、1つのDataFrameにする
df_concat = pd.concat([df_a, df_b])

print(df_a.shape)  # 連結前の行数・列数（１）
print(df_b.shape)  # 連結前の行数・列数（２）
print(df_concat.shape)  # 連結後の行数・列数
```

## グループごとの統計量

- データをグループ分けして何らかの処理を行いたい場合、groupby()が便利です。
- df.groupby(列名).処理のように書くと、指定した列の値のグループごとに処理を行います。
  - たとえば、df.groupby("A").mean()のように書くと、列Aの値ごとにグループ分けして、各グループごとの平均値を計算します。

``` python
import pandas as pd

# データの読み込み
df = pd.read_csv("dataset/physical_measurement_clean.csv")
df

# 列Clubのグループごとに、各列の平均値を計算
mean_df = df.groupby("Club").mean(numeric_only=True)
mean_df

# 列Clubのグループごとに、各列の最大値を計算
max_df = df.groupby("Club").max()
max_df
```

## グラグの描画

折れ線グラフ: plot.line()
棒グラフ: plot.bar()
散布図: plot.scatter()
箱ひげ図: plot.box()

- DataFrameクラスとSeriesクラスは、plot.グラフの種類()とすることでグラフを描画できます。
  - たとえばdf.plot.scatter(x=x軸に使う列名, y=y軸に使う列名)のように書くと、散布図を描画できます。

``` python
import pandas as pd

# データの読み込み
df = pd.read_csv("dataset/physical_measurement_clean.csv")
# x軸が列"Height", y軸が列"Weight"の散布図を描画
df.plot.scatter(x="Height", y="Weight")
```

---
title: DataFrame
---

## DataFrameを構成する要素

DataFrameを構成するのは、インデックス、列名一覧、データ

| |0|1| <- 列名一覧, １要素だと列、カラム,column
|0|A|B|
|0|C|D|　<- A ,B ,C,Dの部分がデータ
 ↑
インデックス、１要素だと行、ロー、row

``` python
import pandas as pd

df = pd.DataFrame([[1, 2], [3, 4]])
```

## DattaFrameの構成要素の取得など

DataFrame型の変数dfの3つの部分は、それぞれ以下のように取得できます。

df.index：インデックス
df.columns：列名一覧
df.to_numpy()：データ

``` python
import pandas as pd

df = pd.DataFrame(
    [["Alice", 17], ["Bob", 24]],
    index=[10, 20],
    columns=["Name", "Age"],
)

df_index = df.index
df_columns = df.columns
df_data = df.to_numpy()
```

## インデックスと列名一覧の変更

- dfのインデックスを変更するには、df.indexに代入します。
- dfの列名一覧を変更するには、df.columnsに代入します。

``` python
import pandas as pd

# indexとcolumnsを指定せずにDataFrameを作成します。
df = pd.DataFrame([["Alice", 17], ["Bob", 24]])

# df.indexを指定します。インデックスが変わることを確認してください。
df.index = [10, 20]

# df.columnsを指定します。列名一覧が変わることを確認してください。
df.columns = ["Name", "Age"]
```

## データの取得と変更

- 特定の行は、df.loc[行名]で、取得・変更できます。
- 特定の列は、df[列名]あるいはdf.loc[:, 列名]で、取得・変更できます。
- 特定の行、特定の列は、df.loc[行名, 列名]で、取得・変更できます。

列名がDataFrameの属性と同名の場合は、df.列名でなくdf[列名]を使うようにしましょう。

たとえば、indexという名前の列があるとします。このときdf.indexとしても、列indexではなくインデックスを指定してしまいます。そのため、indexという列を使うにはdf["index"]と書く必要があります。

また、バグを回避するために、なるべく属性と同じ名前の列を使わない方がよいでしょう。

``` python
import pandas as pd

df = pd.DataFrame(
    [["Alice", 17], ["Bob", 24]],
    index=[10, 20],
    columns=["Name", "Age"],
)

# 行名を指定した行の取得と変更
## 行名が10の行を取得します。
df.loc[10]

## 行名が10の行を変更し、確認します。
df.loc[10] = ["Carol", 18]

# 列名を指定した列の取得
## 列の取得方法その1
df.loc[:, "Age"]

## 列の取得方法その2
df["Age"]

## 列の取得方法その3
df.Age

# 列名を指定した列の変更
## 列の変更方法その1
df.loc[:, "Age"] = [19, 25]

## 列の変更方法その2
df["Age"] = [19, 25]

## 列の変更方法その3
df.Age = [19, 25]

# 行名と列名で指定した部分の取得と変更
## 行名が20で、列名がAgeの列を取得します。
df.loc[20, "Age"]

## 行名が20で、列名がAgeの列を変更し、確認します。
df.loc[20, "Age"] = 26
```

## 行名リストと列名のリスト

df.loc[行名, 列名]による取得や変更だけでなく、下記の書き方をすることで、複数行・複数列にわたるデータの範囲を指定できることを学びました。

- df.loc[行名, 列名のリスト]により、取得や変更ができます。
- df.loc[行名のリスト, 列名]により、取得や変更ができます。
- df.loc[行名のリスト, 列名のリスト]により、取得や変更ができます。
- 複数の列は、df[列名のリスト]あるいはdf.loc[:, 列名のリスト]で、取得や変更ができます。

``` python
import pandas as pd

df = pd.DataFrame(
    [["Alice", 17, "A"], ["Bob", 24, "B"], ["Carol", 29, "C"]],
    index=[10, 20, 30],
    columns=["Name", "Age", "Team"],
)
# 取得　行名が10で、列名がNameとAgeのデータを取得します。
df.loc[10, ["Name", "Age"]]
# 変更　行名が10で、列名がNameとAgeのデータを変更します。
df.loc[10, ["Name", "Age"]] = ["Amanda", 18]
# 行名が10と20で、列名がAgeのデータを取得します。
df.loc[[10, 20], "Age"]
# 行名が10と20で、列名がNameとAgeのデータを変更します。
df.loc[[10, 20], "Age"] = [21, 27]
# 行名が10と20で、列名がNameとAgeのデータを取得します。
df.loc[[10, 20], ["Name", "Age"]]
# 行名が10と20で、列名がNameとAgeのデータを変更します。
df.loc[[10, 20], ["Name", "Age"]] = [["Ada", 19], ["Ben", 28]]
# 列名がNameとTeamのデータを変更し、その結果を確認します。
df[["Name", "Team"]] = [["Dana", "D"], ["Eve", "E"], ["Grace", "G"]]
```

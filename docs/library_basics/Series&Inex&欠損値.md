---
title: Series&Inex&欠損値
---

## Seriesとは

DataFrameを構成する列や行（1次元のデータ構造）
1次元のデータ構造をSeries（シリーズ）といいます。

### Seriesでないもの

- 1つの値
- ２次元のデータ
  - 複数行
  - 複数列

### 取得にかかるコスト

DataFrameの内部の構造は、列で構成されています。
そのため、「列の取得」では新しいSeriesのオブジェクトは生成されず、
比較的小さいコストで処理を行えます。

これに対し、「行の取得」では新しくSeriesのオブジェクトが生成されます。
そのためメモリーを使ったり計算時間がかかったりします。
すなわち「行の取得」は「列の取得」に比べてコストがかかります。

## SeriesとIndexの作成

- Seriesを作成するには、pd.Series(データ, index=インデックス, name=名前)のように書きます。
- Indexを作成するには、pd.Index(データ, name=名前)のように書きます。

``` python
# pd.Series(データ, index=インデックス, name=名前)
sr = pd.Series([17, 24, 29], index=[10, 20, 30], name="Age")

# pd.Index(データ, name=名前)
idx = pd.Index([10, 20, 30], name="MyIndex")
```

``` python
import pandas as pd
# シリーズを作成
sr = pd.Series([17, 24, 29], index=[10, 20, 30], name="Age")
# シリーズに付けた名前を確認する
sr.name
# シリーズのインデックスを確認
sr.index
# インデックスを作成
idx = pd.Index([10, 20, 30], name="MyIndex")
# インデックスに付けた名前を確認する
idx.name
```

## Seriesの要素の取得と変更

- Seriesの変数srの要素は、sr[行名]で取得できます。
- Seriesの変数srの要素は、sr[行名] = 新しい値で変更できます。

``` python
import pandas as pd

sr = pd.Series([17, 24, 29], index=[10, 20, 30], name="Age")

# 行名が20の値を取得する
sr[20]
# 行名が20の値を25に変更します。
sr[20] = 25
# sr[20]は、sr.loc[20]と同じです。
sr.loc[20]
# DataFrame同様、Seriesもilocを使えます。sr.iloc[1]と書くと、行番号1の値を取得できます。
sr.iloc[1]
```

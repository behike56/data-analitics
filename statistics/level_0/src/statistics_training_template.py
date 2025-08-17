
import pandas as pd
import numpy as np
from scipy import stats

# データ読み込み
df = pd.read_csv("statistics_training_sample.csv")

# 記述統計
print("平均年齢:", df['age'].mean())
print("年齢の標準偏差:", df['age'].std())

# t検定（生存者と非生存者の年齢差）
age_survived = df[df['survived'] == 1]['age']
age_not_survived = df[df['survived'] == 0]['age']
t_stat, p_value = stats.ttest_ind(age_survived, age_not_survived, equal_var=False)
print("t検定統計量:", t_stat)
print("p値:", p_value)

# カイ二乗検定（生存と運賃のカテゴリ関係）
df['fare_category'] = pd.qcut(df['fare'], 3, labels=['low', 'medium', 'high'])
contingency_table = pd.crosstab(df['fare_category'], df['survived'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("カイ二乗値:", chi2)
print("p値:", p)

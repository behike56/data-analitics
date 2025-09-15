import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# メタ情報行をスキップし、カラム名を指定してデータを読み込み
df = pd.read_csv(
    '/mnt/data/data.csv',
    encoding='cp932',
    skiprows=5,
    names=['datetime', 'temperature', 'quality', 'homogeneity'],
    parse_dates=['datetime']
)

# 月日ごとの平均気温を算出
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
daily_avg = df.groupby(['month', 'day'])['temperature'].mean().unstack(level='day')

# ヒートマップの作成
fig, ax = plt.subplots(figsize=(12, 8))
cax = ax.imshow(daily_avg, aspect='auto', origin='lower')

# 軸の設定
ax.set_xticks(np.arange(daily_avg.shape[1]))
ax.set_xticklabels(np.arange(1, daily_avg.shape[1] + 1))
ax.set_yticks(np.arange(12))
ax.set_yticklabels(np.arange(1, 13))
ax.set_xlabel('日')
ax.set_ylabel('月')
ax.set_title('2023年東京都 毎日の平均気温ヒートマップ')

# カラーバー
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('平均気温 (℃)')

plt.tight_layout()
plt.show()

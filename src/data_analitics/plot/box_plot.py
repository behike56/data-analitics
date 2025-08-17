import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# データ読み込み
df = pd.read_csv('/mnt/data/Tokyo_Setagaya_Ward_20231_20234_with_price_per_sqm.csv', encoding='cp932')

# 各駅の中央値でソート（降順）
medians = df.groupby('最寄駅：名称')['㎡単価（円/m2）'].median().sort_values(ascending=False)
stations = medians.index.tolist()

# ソートに基づくデータリストを作成
data = [df.loc[df['最寄駅：名称'] == station, '㎡単価（円/m2）'].dropna().values for station in stations]

# 日本語フォント設定
font_path = '/mnt/data/NotoSansJP-Light.ttf'
jp_font = FontProperties(fname=font_path)

# 横向き箱ひげ図を作成（外れ値を×印に変更）
plt.figure(figsize=(8, max(6, len(stations) * 0.2)))
plt.boxplot(
    data,
    vert=False,
    flierprops={'marker': 'x'}
)
plt.yticks(range(1, len(stations) + 1), stations, fontproperties=jp_font)
plt.xlabel('㎡単価（円/m2）', fontproperties=jp_font)
plt.ylabel('最寄駅', fontproperties=jp_font)
plt.title('駅別 ㎡単価の箱ひげ図（外れ値×印）', fontproperties=jp_font)
plt.tight_layout()
plt.show()

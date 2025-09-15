import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# データ読み込み
tables = pd.read_html('/mnt/data/気象庁 _ 過去の梅雨入りと梅雨明け（関東甲信）.html', encoding='utf-8')
df = tables[1]
df.columns = df.columns.get_level_values(1)
df.columns = ['Year', 'Start', 'End', 'PrecipRatio']

# 平年行を除外し、数値年に変換
df = df[df['Year'].str.match(r'^\d+年$')].reset_index(drop=True)
df['Year_num'] = df['Year'].str.replace('年', '', regex=False).astype(int)

# 日付パース
df['Start_clean'] = df['Start'].str.replace('ごろ', '', regex=False)
df['Start_date'] = pd.to_datetime(df['Year_num'].astype(str) + '年' + df['Start_clean'], format='%Y年%m月%d日')
df['End_clean'] = df['End'].str.replace('ごろ', '', regex=False).replace('－', pd.NA)
df['End_date'] = pd.to_datetime(df['Year_num'].astype(str) + '年' + df['End_clean'], format='%Y年%m月%d日', errors='coerce')

# プロット用に年を2000年に統一
df['Start_plot'] = df['Start_date'].apply(lambda d: d.replace(year=2000))
df['End_plot'] = df['End_date'].apply(lambda d: d.replace(year=2000) if not pd.isna(d) else pd.NaT)

# 10年毎の年ラベル作成
min_year = df['Year_num'].min()
max_year = df['Year_num'].max()
start_tick = (min_year // 10) * 10
end_tick = ((max_year + 9) // 10) * 10
yticks = list(range(start_tick, end_tick + 1, 10))

# 図作成
fig, ax = plt.subplots(figsize=(8, 12))
for y, start, end in zip(df['Year_num'], df['Start_plot'], df['End_plot']):
    if pd.isna(end):
        ax.plot(start, y, marker='o', linestyle='')
    else:
        ax.plot([start, end], [y, y], marker='o', linestyle='-')

# 軸設定
ax.set_ylim(max_year + 1, min_year - 1)  # 上に最新年
ax.set_yticks(yticks)
ax.set_yticklabels(yticks)
ax.set_xlabel('日付')
ax.set_ylabel('年')

# x軸範囲と月ラベル
ax.set_xlim(pd.to_datetime('2000-05-01'), pd.to_datetime('2000-08-31'))
xticks = [pd.to_datetime(f'2000-{m:02d}-01') for m in [5, 6, 7, 8]]
ax.set_xticks(xticks)
ax.set_xticklabels(['5月', '6月', '7月', '8月'])

fig.tight_layout()
plt.show()

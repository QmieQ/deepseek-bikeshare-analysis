"""需要两个库：pandas、matplotlib。安装命令：python -m pip install pandas matplotlib"""
from base64 import b64encode
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.mlab import GaussianKDE
import pandas as pd

# 每行是一次骑行。直接从脚本所在的文件夹读取 CSV，双击报告就能看结果。
folder = Path(__file__).resolve().parent
rides = pd.read_csv(folder / "泽西市共享单车骑行记录.csv", parse_dates=["开始时间"])
start = rides["开始时间"]
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "figure.figsize": (8, 3), "axes.unicode_minus": False})
sections = []


def save_chart(title, conclusion):
    # 把当前图存进内存，再嵌入网页。这样交作业时只需交一个 HTML。
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    picture = b64encode(buffer.getvalue()).decode("ascii")
    sections.append(f'<section><h2>{title}</h2><p>{conclusion}</p>'
                    f'<img alt="{title}" src="data:image/png;base64,{picture}"></section>')
    plt.close()


# 1. 每天多少次？按天分组，size() 数出每一天有多少条骑行记录。
daily = rides.resample("D", on="开始时间").size()
daily.plot.line(xlabel="日期", ylabel="骑行次数", marker=".", color="#16766b")
save_chart("每日骑行量", f'{daily.idxmax():%m月%d日}最多，共 {daily.max():,} 次；'
           f'{daily.idxmin():%m月%d日}最少，共 {daily.min():,} 次。')

# 2. 几点最忙？从开始时间取出小时，统计 0 到 23 点各有多少次出发。
hourly = start.dt.hour.value_counts().reindex(range(24), fill_value=0)
hourly.plot.bar(xlabel="出发小时", ylabel="骑行次数", rot=0, color="#16766b")
save_chart("一天中的骑行高峰", f"{hourly.idxmax()} 点至 {hourly.idxmax()+1} 点最忙，"
           f"全月这个时段共有 {hourly.max():,} 次出发。")

# 3. 工作日和周末差多少？周一到周五算工作日，每一天作为一个样本画箱线图。
work = daily[daily.index.dayofweek < 5]   # 周一是 0，周五是 4
weekend = daily[daily.index.dayofweek >= 5]
plt.boxplot([work, weekend])
plt.xticks([1, 2], ["工作日", "周末"])
plt.ylabel("每天骑行次数")
save_chart("工作日与周末", f"工作日日均 {work.mean():.0f} 次，周末日均 {weekend.mean():.0f} 次。"
           "箱内横线表示各组的中位数。")

# 4. 哪些站最热门？按起点站数出发次数，只取前十，再让第一名显示在最上面。
stations = rides["起点站名称"].value_counts().head(10)
stations.sort_values().plot.barh(xlabel="出发次数", ylabel="", color="#16766b")
save_chart("出发次数最多的十个站点", f"{stations.index[0]}居首，共 {stations.iloc[0]:,} 次出发。")

# 5. 一趟骑多久？秒除以 60 得到分钟。只画较短的 99%，避免超长记录挤扁横轴。
minutes = rides["骑行时长"] / 60
limit = minutes.quantile(0.99)
shown = minutes[minutes <= limit]
_, edges, _ = plt.hist(shown, bins=40, density=True, color="#87cbb7")
kde = GaussianKDE(shown)  # 根据时长估计平滑曲线，和直方图共用密度刻度
plt.plot(edges, kde(edges), color="#155956")
plt.xlabel("骑行时长（分钟）")
plt.ylabel("概率密度")
save_chart("单次骑行时长", f"全部骑行的中位数为 {minutes.median():.1f} 分钟。"
           f"图中展示 {limit:.1f} 分钟以内的记录，曲线表示时长的分布密度。")

# 6. 星期几、几点最忙？先数每格的出发次数，再除以这个星期几在当月出现的天数。
counts = pd.crosstab(start.dt.dayofweek, start.dt.hour)
heat = counts.div(daily.index.dayofweek.value_counts(), axis=0).sort_index()
heat.index = list("一二三四五六日")
plt.imshow(heat, cmap="YlGnBu", aspect="auto")
plt.xticks(range(0, 24, 2))
plt.yticks(range(7), ["周" + day for day in heat.index])
plt.xlabel("出发小时")
plt.colorbar(label="平均出发次数")
weekday, hour = heat.stack().idxmax()
save_chart("一周中的骑行高峰", f"周{weekday} {hour} 点最忙，平均每个周{weekday}的这个小时有 "
           f"{heat.loc[weekday, hour]:.1f} 次出发。")

# 7. 哪类用户骑得多？按用户类型统计骑行次数，autopct 把比例标在饼图上。
users = rides["用户类型"].value_counts()
users.plot.pie(autopct="%.1f%%", ylabel="", colors=["#87cbb7", "#f4bc70"], startangle=90)
save_chart("各类用户贡献的骑行次数", f"{users.index[0]}贡献了 {users.iloc[0]:,} 次骑行，"
           f"占全部骑行的 {users.iloc[0] / len(rides):.1%}。")

# 把标题、日期和七组图文拼成一个网页，保存在 CSV 旁边。
html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>泽西市共享单车骑行报告</title>
<style>body{{max-width:850px;margin:36px auto;padding:0 18px;font:16px/1.65 sans-serif;color:#193537}}
h1,h2{{line-height:1.3}}section{{border-top:1px solid #cad9d5;margin-top:28px;break-inside:avoid}}
img{{max-width:100%;height:auto}}</style><h1>泽西市共享单车骑行报告</h1>
<p>{start.min():%Y年%m月%d日}至{start.max():%m月%d日} · 共 {len(rides):,} 次骑行</p>
{''.join(sections)}<footer>数据来源：泽西市共享单车骑行记录.csv</footer></html>"""
report = folder / "骑行报告.html"
report.write_text(html, encoding="utf-8")
print(report)

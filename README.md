# DeepSeek 共享单车数据分析助手

这是一个命令行分析项目。用户用自然语言提出问题，DeepSeek 制定分析方案并选择工具，Python 在本地 CSV 上计算统计结果。模型可以根据结果回答问题，也可以把分析保存为 HTML 报告。

## 报告示例

- [首尔与泽西市共享单车数据报告](examples/首尔与泽西市共享单车数据报告.html)
- [泽西市骑行可视化报告](examples/泽西市骑行可视化示例.html)

克隆仓库后可用浏览器打开这两个 HTML 文件。双城报告使用首尔 8,760 条逐小时租赁与天气记录，以及泽西市 19,676 条骑行记录。两城的数据粒度不同，报告分别解释统计口径。

## 快速开始

以下命令以 Windows PowerShell 和 Python 3.13 为例。在刚克隆的项目根目录中执行。仓库自带 `examples/data/` 下的合成数据，供试跑使用。已经准备好同名正式数据时，跳过复制样例的命令。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item -Path .\examples\data\*.csv -Destination .\data\
```

无需 API 密钥即可验证本地统计。

```powershell
.\.venv\Scripts\python.exe -c "from ana import check_data_quality; print(check_data_quality('seoul'))"
```

使用 DeepSeek 时，在本地终端将密钥配置为环境变量 `dskey_shixi`，然后启动入口。程序启动后先运行内置双城分析任务，再接受后续问题。模型会收到问题和本轮返回的工具结果。

```powershell
.\.venv\Scripts\python.exe main.py
```

也可以独立运行 `python data/骑行分析.py`，用 Matplotlib 生成 `data/骑行报告.html`。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `main.py` | DeepSeek API 调用、分析规划、工具调度、上下文摘要与 HTML 导出 |
| `ana.py` | 14 个分析工具，涵盖数据质量、分组比较、时间趋势、异常值、相关性与线性回归 |
| `data/骑行分析.py` | 用 pandas 和 Matplotlib 生成泽西市骑行图表 |
| `examples/` | 合成 CSV 和基于完整数据生成的示例报告 |

## 数据格式

使用自己的数据时，按 `examples/data/` 中的中文表头整理成 UTF-8 CSV，存放到 `data/`。

- `首尔共享单车租赁与天气数据.csv` 使用日/月/年格式的日期和单独的小时列。原始数据见 [UCI Seoul Bike Sharing Demand](https://doi.org/10.24432/C5F62R)（Easwaramoorthy，2020，CC BY 4.0）。
- `泽西市共享单车骑行记录.csv` 每行对应一次骑行，包含起止时间、站点和用户类别等字段。历史数据见 [Citi Bike System Data](https://citibikenyc.com/system-data)（泽西市 2019 年 1 月）。数据使用条件见 [Citi Bike Data Sharing Policy](https://citibikenyc.com/data-sharing-policy)。

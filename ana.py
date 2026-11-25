"""本地数据分析工具：Python 计算完整数据，向 AI 返回 JSON 统计结果或数据预览。"""

import json
from collections.abc import Callable
from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np


# 路径相对于本文件，换一个终端工作目录也可以运行。
DATA_DIR = Path(__file__).resolve().parent / "data"
FILES = {
    "seoul": "首尔共享单车租赁与天气数据.csv",
    "jersey": "泽西市共享单车骑行记录.csv",
}


@lru_cache(maxsize=2)
def _load(dataset):
    """按 seoul/jersey 读取对应 CSV 并缓存，重复分析时复用已加载的数据。"""
    return pd.read_csv(DATA_DIR / FILES[dataset])


def _json(value):
    """把分析结果转成 JSON 文本，统一处理 NumPy 标量、缺失值和非有限数。

    NaN、正负无穷大、pd.NA 和 pd.NaT 输出为 null，便于 AI 解读和网页使用。
    """
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if x is pd.NA or x is pd.NaT:
            return None
        return x
    return json.dumps(clean(value), ensure_ascii=False, default=str, allow_nan=False)


def _dates(df, dataset, field=None):
    """解析日期列；首尔默认合并“日期”和“小时”，泽西市默认使用“开始时间”。

    首尔“日期”按日/月/年解析；无法解析的值记为 NaT，供质量检查或趋势统计识别。
    """
    field = field or ("日期" if dataset == "seoul" else "开始时间")
    if field not in df:
        raise ValueError(f"不存在日期字段：{field}")
    # 首尔的 01/12/2017 是12月1日，不是1月12日。
    fmt = "%d/%m/%Y" if dataset == "seoul" and field == "日期" else "mixed"
    values = pd.to_datetime(df[field], format=fmt, errors="coerce")
    if dataset == "seoul" and field == "日期":
        values = values + pd.to_timedelta(df["小时"], unit="h")
    return values


def _integer(value, name, minimum=1, maximum=100):
    """校验分页数量、偏移量等整数参数的范围；布尔值也视为无效参数。"""
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须为{minimum}至{maximum}的整数")


def _fields(df, *columns):
    """检查指定字段是否存在，跳过未传入的 None；字段错误时提示先查看数据。"""
    for column in columns:
        if column is not None and column not in df:
            raise ValueError(f"不存在字段：{column}，请先 view_data 查看字段")


def _frame(dataset, filters=None):
    """读取数据并应用全部筛选条件，返回筛选后的表和本次分析范围。

    多个 filters 条件按“且”组合，支持比较、集合、区间、文本包含和空值判断。
    between 包含两端，contains 按字面包含匹配；字符串日期不自动转成日期比较。
    不修改原 CSV 或缓存；范围信息包含数据集、原始行数、参与行数和筛选条件。
    """
    if dataset not in FILES:
        raise ValueError("dataset 必须为 seoul 或 jersey")
    if filters is not None and not isinstance(filters, list):
        raise ValueError("filters 必须为条件数组")
    df = _load(dataset)
    original_rows = len(df)
    for rule in filters or []:
        if not isinstance(rule, dict) or "column" not in rule or "op" not in rule:
            raise ValueError("每条筛选条件必须包含 column 和 op")
        field, op = rule["column"], rule["op"]
        _fields(df, field)
        series = df[field]
        if op in ("is_null", "not_null"):
            mask = series.isna() if op == "is_null" else series.notna()
        else:
            if "value" not in rule:
                raise ValueError(f"筛选操作 {op} 需要 value")
            value = rule["value"]
            if op in ("in", "not_in", "between"):
                if not isinstance(value, list) or (op == "between" and len(value) != 2):
                    raise ValueError("in/not_in 的 value 使用数组；between 使用[下界, 上界]")
            try:
                if op == "eq": mask = series.eq(value)
                elif op == "ne": mask = series.ne(value) & series.notna()
                elif op == "gt": mask = series.gt(value)
                elif op == "ge": mask = series.ge(value)
                elif op == "lt": mask = series.lt(value)
                elif op == "le": mask = series.le(value)
                elif op == "in": mask = series.isin(value)
                elif op == "not_in": mask = ~series.isin(value) & series.notna()
                elif op == "between": mask = series.between(value[0], value[1], inclusive="both")
                elif op == "contains":
                    if not isinstance(value, str):
                        raise ValueError("contains 的 value 必须为字符串")
                    mask = series.astype("string").str.contains(value, regex=False, na=False)
                else: raise ValueError(f"不支持的筛选操作：{op}")
            except TypeError as error:
                raise ValueError(f"字段 {field} 的筛选值与数据类型不兼容") from error
        df = df.loc[mask.fillna(False)]
    scope = {"数据集": dataset, "原始行数": original_rows, "本次参与行数": len(df), "筛选条件": filters or []}
    return df, scope


def _numeric(df, column):
    """校验数值字段，并将正负无穷大转成缺失值，供统计时排除。"""
    _fields(df, column)
    if not column or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError("请选择数值 column")
    return df[column].replace([np.inf, -np.inf], np.nan)


def analyze_data(dataset, action, column=None, group_by=None, method="mean",
                 filters=None, date_column=None, period="day", limit=30,
                 bins=10, correlation_method="pearson"):
    """兼容旧代码的统一分析入口，根据 action 选择分析操作并返回 JSON 文本。

    overview/quality/describe 分别查看概况、质量和描述统计；group/compare 做分组统计；
    time/frequency 分析时间变化或类别频数；distribution/outliers 检查分布或异常候选；
    correlation 计算相关性。独立工具也会复用本函数中的部分计算逻辑。
    filters 先筛选完整数据，limit 只限制结果展示；参数错误由主程序反馈给 AI。
    """
    _integer(limit, "limit")
    df, scope = _frame(dataset, filters)
    _fields(df, column, group_by)
    if df.empty:
        return _json({**scope, "提示": "筛选后没有数据，请调整条件。"})
    def output(result):
        return _json({**scope, "结果": result})

    # 数据质量：检查重复、缺失、非有限数、常量列及日期范围，不自动删除记录。
    if action == "quality":
        dates = _dates(df, dataset, date_column)
        return output({
            "重复行数": int(df.duplicated().sum()),
            "字段质量": [{"字段": c, "缺失数": int(df[c].isna().sum()),
                         "非有限数": int(df[c].isin([np.inf, -np.inf]).sum()) if pd.api.types.is_numeric_dtype(df[c]) else 0,
                         "缺失比例": float(df[c].isna().mean()),
                         "不同值数": int(df[c].nunique()),
                         "常量列": bool(df[c].nunique() <= 1)} for c in df],
            "时间起点": dates.min(), "时间终点": dates.max(),
            "日期缺失或解析失败数": int(dates.isna().sum()),
            "说明": "不自动删除重复行或异常值；骑行时长等字段单位需根据数据说明确认。",
        })

    # 类别频数：按数量降序返回类别及占比，明确统计未展示的记录数。
    if action == "frequency":
        if not column:
            raise ValueError("frequency 需要 column")
        counts = df[column].value_counts(dropna=False)
        return output({"类别总数": len(counts), "仅展示前": limit,
                       "未展示记录数": int(counts.iloc[limit:].sum()),
                       "分布": [{"类别": None if pd.isna(k) else k, "数量": int(v),
                                "占全部记录比例": float(v / len(df))}
                               for k, v in counts.head(limit).items()]})

    # 数值分布与异常检测共用基础统计；前者补充直方图，后者使用 1.5 倍 IQR 阈值。
    if action in ("distribution", "outliers"):
        values = _numeric(df, column).dropna()
        if values.empty:
            return output({"有效数": 0, "缺失或非有限数": len(df), "提示": "没有有限数值"})
        details = {"有效数": len(values), "缺失或非有限数": len(df) - len(values),
                   "均值": values.mean(), "中位数": values.median(),
                   "标准差": values.std(),
                   "分位数": values.quantile([0, .01, .05, .25, .5, .75, .95, .99, 1]).to_dict()}
        if action == "distribution":
            _integer(bins, "bins", maximum=50)
            counts, edges = np.histogram(values, bins=bins)
            details["直方图"] = [{"下界": edges[i], "上界": edges[i+1], "数量": int(n)}
                                 for i, n in enumerate(counts)]
            details["区间规则"] = "左闭右开，最后一个区间包含右端点"
        else:
            q1, q3 = values.quantile([.25, .75])
            low, high = q1 - 1.5 * (q3-q1), q3 + 1.5 * (q3-q1)
            unusual = values[(values < low) | (values > high)]
            details.update({"IQR下界": low, "IQR上界": high, "异常候选数": len(unusual),
                            "异常候选比例": len(unusual)/len(values),
                            "说明": "1.5倍IQR标记候选，不表示数据错误，不自动删除。"})
        return output(details)

    # 时间趋势按日期或周期分组；组间比较按类别分组，两者都可统计记录数或数值指标。
    if action in ("time", "compare"):
        if action == "time":
            dates = _dates(df, dataset, date_column)
            if period in ("day", "week", "month"):
                key = dates.dt.to_period({"day": "D", "week": "W-SUN", "month": "M"}[period]).dt.start_time
            elif period == "hour": key = dates.dt.hour
            elif period == "weekday": key = dates.dt.dayofweek
            else: raise ValueError("period 必须为 day/week/month/hour/weekday")
            excluded = int(dates.isna().sum())
        else:
            if not group_by: raise ValueError("compare 需要 group_by")
            key = df[group_by]
            excluded = 0
        grouped = df.groupby(key, dropna=action == "time")
        sizes = grouped.size()
        if column:
            values = _numeric(df, column)
            numbers = values.groupby(key, dropna=action == "time")
            table = numbers.agg(["count", "mean", "median", "std", "min", "max"])
            table["sum"] = numbers.sum(min_count=1)
            table["记录数"] = sizes
            table["缺失或非有限数"] = sizes - table["count"]
        else:
            table = sizes.to_frame("记录数")
        if action == "compare": table = table.sort_values("记录数", ascending=False)
        rows = [{"分组": None if pd.isna(k) else k, **v} for k, v in table.head(limit).to_dict("index").items()]
        return output({"总组数": len(table), "已返回组数": len(rows), "是否截断": len(table)>limit,
                       "日期无效排除数": excluded, "各组": rows,
                       "说明": "time按时间升序；weekday为0周一至6周日；week以周一开始。无记录时段不补零。count为有限数值数，记录数含数值缺失行。"})

    # 数据概况：返回行数、字段类型和缺失计数，帮助 AI 选择后续分析字段。
    if action == "overview":
        return output({
            "总行数": len(df),
            "字段类型": df.dtypes.astype(str).to_dict(),
            "缺失值数量": df.isna().sum().to_dict(),
            "说明": "先quality检查质量。可用time分析趋势、frequency分析站点用户、compare比较分组、distribution/outliers检查分布和异常。",
        })

    # 描述统计：数值列给出均值和分位数等，类别列给出不同值数、最高频类别等。
    if action == "describe":
        target = df[[column]] if column else df
        target = target.replace([np.inf, -np.inf], np.nan)
        return output(target.describe(include="all").to_dict())

    # 分组单项统计：每组只计算 method 指定的指标；count 表示组内全部记录数。
    if action == "group":
        if method not in ("mean", "sum", "min", "max", "median", "count", "std", "var"):
            raise ValueError("不支持的统计方法")
        if not group_by:
            raise ValueError("分组统计需要 group_by 字段")
        grouped = df.groupby(group_by, dropna=False)
        sizes = grouped.size()
        if method == "count":
            # count 在本工具中表示每组记录数，不需要 column。
            result = grouped.size()
        else:
            values = _numeric(df, column)
            grouped = values.groupby(df[group_by], dropna=False)
            # sum 保留全空组为缺失值，避免误报为零。
            result = (grouped.sum(min_count=1) if method == "sum"
                      else grouped.agg(method))
        details = {"统计方法": method, "分组结果": result.head(limit).to_dict(),
                   "总组数": len(result), "已返回组数": min(len(result), limit),
                   "是否截断": len(result) > limit,
                   "分组记录数": sizes.head(limit).to_dict()}
        if method != "count":
            counts = grouped.count()
            details.update({"分组有效数": counts.head(limit).to_dict(),
                            "分组缺失或非有限数": (sizes-counts).head(limit).to_dict()})
        return output(details)

    # 相关性：仅使用两个字段均为有限数的配对记录，常量列或样本不足时返回 null。
    if action == "correlation":
        if not column or not group_by or column == group_by:
            raise ValueError("请用 column 和 group_by 指定两个不同数值字段")
        pair = df[[column, group_by]]
        if not all(pd.api.types.is_numeric_dtype(t) for t in pair.dtypes):
            raise ValueError("相关性分析需要两个数值字段")
        valid = pair.dropna()
        if correlation_method not in ("pearson", "spearman"):
            raise ValueError("correlation_method 必须为 pearson 或 spearman")
        valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
        left, right = valid.iloc[:, 0], valid.iloc[:, 1]
        if correlation_method == "spearman":
            left, right = left.rank(), right.rank()
        value = left.corr(right) if len(valid) >= 2 and left.nunique()>1 and right.nunique()>1 else float("nan")
        return output({
            "方法": correlation_method,
            "有效配对记录数": len(valid),
            "相关系数": None if pd.isna(value) else float(value),
            "说明": "相关性不代表因果；常量列或有效记录不足时无法计算。",
        })

    raise ValueError("未知分析操作")


# 旧统一入口的工具声明：提供给模型的名称、用途和参数格式，不负责执行计算。
ANALYZE_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_data",
        "description": (
            "对本地完整数据统计。seoul 为首尔租赁天气数据，jersey 为泽西市骑行记录。"
            "先 overview 查看字段，再分析。数值统计跳过缺失值，分组保留缺失类别。"
            "相关性不是因果关系，编号和类别编码不应机械解释为连续变量。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "enum": ["seoul", "jersey"]},
                "action": {"type": "string", "enum": ["overview", "quality", "describe", "group", "compare", "time", "frequency", "distribution", "outliers", "correlation"],
                           "description": "overview字段; quality质量和时间范围; describe统计; group单指标; compare组间多指标; time时间分组; frequency类别排名; distribution直方图; outliers异常候选; correlation相关性"},
                "column": {"type": "string", "description": "要统计的原始字段名"},
                "group_by": {"type": "string", "description": "分组字段；correlation 时为第二个数值字段"},
                "method": {"type": "string", "enum": ["mean", "sum", "min", "max", "median", "count"],
                           "description": "分组方法，默认 mean；count 为记录数，无需 column"},
                "date_column": {"type": "string", "description": "time/quality日期列，默认首尔日期+小时，泽西市开始时间"},
                "period": {"type": "string", "enum": ["day", "week", "month", "hour", "weekday"], "description": "time分组粒度，默认day。column不填统计记录数，填写则返回该数值列各项统计"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "最多展示多少组，默认30。超限时明确标注截断，可改用月汇总"},
                "bins": {"type": "integer", "minimum": 1, "maximum": 50, "description": "distribution直方图区间数，默认10"},
                "correlation_method": {"type": "string", "enum": ["pearson", "spearman"]},
                "filters": {"type": "array", "description": "可选筛选，全部条件同时满足，例如仅分析正常运营记录。不改变原数据", "items": {
                    "type": "object", "properties": {
                        "column": {"type": "string"},
                        "op": {"type": "string", "enum": ["eq", "ne", "gt", "ge", "lt", "le", "in"]},
                        "value": {"description": "与字段类型一致的值；in使用数组", "anyOf": [{"type": "string"}, {"type": "number"}, {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}]}}]},
                    }, "required": ["column", "op", "value"]}},
            },
            "required": ["dataset", "action"],
        },
    },
}


# 独立工具：AI 可直接选择目的，不必记住 analyze_data 的 action 约定。
def view_data(dataset, columns=None, filters=None, sort_by=None, ascending=True,
              offset=0, limit=10):
    """查看数据字段、缺失情况和一页原始记录，帮助确定后续分析对象。
    dataset 指定城市数据；filters 先筛选，columns 选择返回字段。
    sort_by 指定排序字段，ascending 控制升降序，缺失值排在末尾。
    offset 从0开始翻页；limit 为1至100，只限制预览行数。
    返回 JSON 文本，含字段类型、缺失数、记录、截断标记和下一页偏移量。
    预览保留缺失记录；它只是完整筛选数据的一页，不代表整体统计。
    """
    _integer(limit, "limit")
    _integer(offset, "offset", minimum=0, maximum=2**63-1)
    if not isinstance(ascending, bool):
        raise ValueError("ascending 必须为布尔值")
    df, scope = _frame(dataset, filters)
    if columns is not None and (not isinstance(columns, list) or not columns
                                or not all(isinstance(c, str) for c in columns)
                                or len(set(columns)) != len(columns)):
        raise ValueError("columns 必须为不重复的非空字段名数组")
    _fields(df, *(columns or []), sort_by)
    if sort_by:
        df = df.sort_values(sort_by, ascending=ascending, kind="stable", na_position="last")
    target = df[columns] if columns else df
    page = target.iloc[offset:offset+limit]
    next_offset = offset + len(page)
    return _json({**scope, "结果": {
        "字段类型": df.dtypes.astype(str).to_dict(),
        "缺失值数量": df.isna().sum().to_dict(),
        "返回字段": list(target.columns), "偏移量": offset,
        "已返回行数": len(page), "是否截断": offset > 0 or next_offset < len(df),
        "下一页偏移量": next_offset if next_offset < len(df) else None,
        "记录": page.to_dict("records"),
        "说明": "预览不代表完整统计；所有分析工具均在完整筛选结果上计算。",
    }})


def mean_value(dataset, column, group_by=None, filters=None, limit=30):
    """计算指定数值字段的整体平均值，也可同时查看各组平均值。
    dataset 指定数据，column 为数值字段，filters 筛选参与记录。
    group_by 可指定类别字段；缺失类别保留为一组。
    返回 JSON 文本，含均值、有效数、缺失或非有限数及可选分组结果。
    缺失值和正负无穷不参与计算，无有效数时均值为 null。
    limit 为1至100，只限制分组展示数量，整体均值仍使用全部筛选记录。
    """
    _integer(limit, "limit")
    df, scope = _frame(dataset, filters)
    _fields(df, group_by)
    values = _numeric(df, column)
    result = {"字段": column, "均值": values.mean(), "有效数": int(values.count()),
              "缺失或非有限数": int(values.isna().sum())}
    if group_by:
        grouped = values.groupby(df[group_by], dropna=False)
        table = grouped.agg(["mean", "count", "size"])
        result.update({
            "分组字段": group_by, "总组数": len(table), "是否截断": len(table) > limit,
            "各组": [{"分组": None if pd.isna(key) else key, "均值": item["mean"],
                      "有效数": int(item["count"]), "记录数": int(item["size"]),
                      "缺失或非有限数": int(item["size"] - item["count"])}
                     for key, item in table.head(limit).to_dict("index").items()],
        })
    return _json({**scope, "结果": result})


def extract_categories(dataset, column, filters=None, offset=0, limit=30):
    """提取某个字段的类别及其频数，了解类别组成与排名。
    dataset 指定数据，column 为分类字段，filters 筛选参与记录。
    offset 从0开始对类别翻页，limit 为1至100，类别按频数降序排列。
    返回 JSON 文本，含类别、数量、占筛选后全部记录比例及截断和翻页信息。
    None、NaN 和 pd.NA 统一为一个缺失类别，比例分母包含未展示类别。
    此工具返回类别统计；提取某类的原始记录请使用 view_data 的 filters。
    """
    _integer(limit, "limit")
    _integer(offset, "offset", minimum=0, maximum=2**63-1)
    df, scope = _frame(dataset, filters)
    if not isinstance(column, str) or not column:
        raise ValueError("分类提取需要非空 column 字段名")
    _fields(df, column)
    # object列可能混合None、NaN和pd.NA，统一为一个缺失类别。
    counts = df[column].mask(df[column].isna()).value_counts(dropna=False)
    page = counts.iloc[offset:offset+limit]
    next_offset = offset + len(page)
    return _json({**scope, "结果": {
        "字段": column, "类别总数": len(counts), "已返回类别数": len(page),
        "偏移量": offset, "是否截断": offset > 0 or next_offset < len(counts),
        "下一页偏移量": next_offset if next_offset < len(counts) else None,
        "未展示记录数": len(df) - int(page.sum()),
        "分布": [{"类别": None if pd.isna(key) else key, "数量": int(count),
                  "占全部记录比例": count / len(df)} for key, count in page.items()],
        "说明": "按频数降序；缺失值作为单独类别。提取某类记录可用 view_data 的 filters。",
    }})


def describe_data(dataset, column=None, filters=None):
    """生成整体数据或指定字段的描述统计，快速了解数值与类别特征。
    dataset 指定数据，filters 筛选记录；column 不填时统计全部字段。
    返回 JSON 文本，数值列含计数、均值、标准差及分位数等，类别列含类别摘要。
    缺失和正负无穷不参与数值统计；标准差使用样本公式（ddof=1）。
    筛选后无数据时返回提示；无法计算的统计量序列化为 null。
    """
    return analyze_data(dataset, "describe", column=column, filters=filters)


def check_data_quality(dataset, filters=None, date_column=None):
    """检查数据质量，包括重复行、缺失、非有限数、常量列和日期解析情况。
    dataset 指定数据，filters 限定检查范围，date_column 可覆盖默认日期字段。
    默认首尔使用日期加小时，泽西市使用开始时间，汇总可解析记录的时间范围。
    返回 JSON 文本，含重复行数、逐字段质量、时间起止及日期无效数量。
    只报告问题，不删除重复行、不填补缺失，也不自动认定异常值是错误。
    """
    return analyze_data(dataset, "quality", filters=filters, date_column=date_column)


def group_statistics(dataset, group_by, column=None, method="mean", filters=None, limit=30):
    """按类别分组计算一个统计指标，便于比较不同类别。
    dataset 和 filters 确定数据范围，group_by 指定分组字段，缺失类别保留。
    method 可选均值、总和、最小、最大、中位数、计数、标准差或方差。
    count 统计每组记录数，无需 column；其他方法要求数值 column 并排除非有限数。
    返回 JSON 文本，含分组结果、记录数、有效数和截断信息；limit 只限制展示组数。
    标准差与方差使用样本公式（ddof=1）；全空组的总和保留为 null。
    """
    return analyze_data(dataset, "group", column=column, group_by=group_by,
                        method=method, filters=filters, limit=limit)


def compare_groups(dataset, group_by, column=None, filters=None, limit=30):
    """同时比较各组的多项统计，按组内记录数从多到少返回。
    dataset 和 filters 确定范围，group_by 指定类别，缺失类别保留为一组。
    column 不填时只统计记录数；指定数值列后计算有效数、均值、中位数等指标。
    返回 JSON 文本，含各组统计、总组数及截断信息；limit 为1至100，只限制展示。
    数值统计排除缺失和正负无穷，标准差使用样本公式（ddof=1）。
    记录数包含数值缺失行，不能直接当作该指标的有效样本数。
    """
    return analyze_data(dataset, "compare", column=column, group_by=group_by,
                        filters=filters, limit=limit)


def time_trend(dataset, column=None, period="day", date_column=None, filters=None, limit=30):
    """按时间汇总记录数或数值指标，观察长期趋势与日内、周内周期。
    dataset 和 filters 确定范围，date_column 可覆盖默认的城市日期字段。
    period 的 day/week/month 为时间段，hour/weekday 汇总跨日期的小时或星期周期。
    column 不填时统计记录数，否则返回有限数值的均值等统计，标准差使用 ddof=1。
    返回 JSON 文本，含时间升序的各组、无效日期排除数及 limit 展示截断信息。
    NaT 不参与分组，不为无记录时段补零；周从周一开始，weekday 的0至6表示周一至周日。
    """
    return analyze_data(dataset, "time", column=column, period=period,
                        date_column=date_column, filters=filters, limit=limit)


def value_distribution(dataset, column, bins=10, filters=None):
    """分析数值字段的分布，返回直方图和常用描述统计。
    dataset 指定数据，column 为数值字段，filters 筛选参与记录。
    bins 为1至50的等宽直方图区间数，默认10；使用全部有限数值计算。
    返回 JSON 文本，含有效数、排除数、均值、中位数、样本标准差和分位数。
    区间左闭右开，最后一个区间包含右端点；标准差使用 ddof=1。
    缺失和正负无穷不参与计算，无有限数值时返回提示，不生成直方图。
    """
    return analyze_data(dataset, "distribution", column=column, bins=bins, filters=filters)


def detect_outliers(dataset, column, filters=None):
    """用四分位距识别数值字段的异常候选，评估极端值的数量与比例。
    dataset 指定数据，column 为数值字段，filters 限定分析范围。
    以 Q1-1.5*IQR 和 Q3+1.5*IQR 为边界，严格落在边界外的值记为候选。
    返回 JSON 文本，含边界、候选数和比例，以及有效数、分位数等统计。
    缺失和非有限数不参与计算；候选不等于数据错误，工具不会删除任何记录。
    """
    return analyze_data(dataset, "outliers", column=column, filters=filters)


def correlation_analysis(dataset, column, other_column, method="pearson", filters=None):
    """衡量两个数值字段的相关程度，辅助探索变量之间的关联。
    dataset 和 filters 确定数据范围，column 与 other_column 必须是不同数值字段。
    method 可选 pearson 线性相关或 spearman 秩相关，默认 pearson。
    返回 JSON 文本，含相关系数、方法和两个字段同时为有限数的有效配对数。
    缺失及正负无穷成对排除；有效配对不足2条或任一字段为常量时系数为 null。
    相关性不代表因果，编号和类别编码的系数也需要结合字段含义解释。
    """
    return analyze_data(dataset, "correlation", column=column, group_by=other_column,
                        correlation_method=method, filters=filters)


def quantile_analysis(dataset, column, quantiles=None, filters=None):
    """计算自选分位点，用于了解数值阈值、尾部和极端水平。
    dataset 指定数据，column 为数值字段，filters 筛选参与记录。
    quantiles 为1至20个0至1的分位点，默认包含 P1、P5、P25、P50、P75、P95、P99。
    返回 JSON 文本，含各分位数、有效数及缺失或非有限数，采用线性插值。
    0和1分别对应最小值与最大值；无有限数值时各分位数为 null。
    """
    quantiles = [.01, .05, .25, .5, .75, .95, .99] if quantiles is None else quantiles
    if (not isinstance(quantiles, list) or not 1 <= len(quantiles) <= 20
            or any(isinstance(q, bool) or not isinstance(q, (int, float))
                   or not np.isfinite(q) or not 0 <= q <= 1 for q in quantiles)):
        raise ValueError("quantiles 必须为1至20个0至1之间的数值")
    df, scope = _frame(dataset, filters)
    values = _numeric(df, column).dropna()
    return _json({**scope, "结果": {
        "字段": column, "有效数": len(values), "缺失或非有限数": len(df)-len(values),
        "分位数": values.quantile(quantiles).to_dict(),
        "说明": "使用线性插值分位数，0为最小值，1为最大值。",
    }})


def cross_tabulate(dataset, row, column, normalize="none", filters=None, limit=20,
                   column_limit=20):
    """对两个类别字段生成交叉计数或占比，观察类别之间的组合分布。
    dataset 和 filters 确定范围，row 与 column 为两个不同字段，缺失保留为类别。
    normalize 可选 none 原始计数、all 全体占比、row 行内占比或 column 列内占比。
    行列分别按频数降序，limit 和 column_limit 各为1至50，只限制展示类别数。
    返回 JSON 文本，含行列类别、矩阵、原始计数合计及未展示记录和截断信息。
    占比分母包含未展示类别；仅构建展示矩阵，避免高基数字段生成巨型矩阵。
    """
    _integer(limit, "limit", maximum=50)
    _integer(column_limit, "column_limit", maximum=50)
    if normalize not in ("none", "all", "row", "column"):
        raise ValueError("normalize 必须为 none/all/row/column")
    if not all(isinstance(field, str) and field for field in (row, column)):
        raise ValueError("交叉表需要非空 row 和 column 字段名")
    if row == column:
        raise ValueError("交叉表需要两个不同的类别字段")
    df, scope = _frame(dataset, filters)
    _fields(df, row, column)
    row_codes, row_labels = pd.factorize(df[row], use_na_sentinel=False)
    col_codes, col_labels = pd.factorize(df[column], use_na_sentinel=False)
    row_totals, col_totals = np.bincount(row_codes), np.bincount(col_codes)
    chosen_rows = np.argsort(-row_totals, kind="stable")[:limit]
    chosen_cols = np.argsort(-col_totals, kind="stable")[:column_limit]
    counts = pd.DataFrame({"r": row_codes, "c": col_codes}).value_counts(sort=False).to_dict()
    matrix, shown = [], 0
    for r in chosen_rows:
        cells = []
        for c in chosen_cols:
            count = int(counts.get((r, c), 0))
            shown += count
            denominator = {"all": len(df), "row": row_totals[r], "column": col_totals[c]}.get(normalize)
            cells.append(count / denominator if denominator is not None else count)
        matrix.append(cells)
    return _json({**scope, "结果": {
        "行字段": row, "列字段": column, "归一化": normalize,
        "行类别总数": len(row_labels), "列类别总数": len(col_labels),
        "行类别": [None if pd.isna(row_labels[r]) else row_labels[r] for r in chosen_rows],
        "列类别": [None if pd.isna(col_labels[c]) else col_labels[c] for c in chosen_cols],
        "矩阵": matrix, "行合计": row_totals[chosen_rows].tolist(),
        "列合计": col_totals[chosen_cols].tolist(), "展示单元格记录数": shown,
        "未展示记录数": len(df)-shown,
        "是否截断": len(row_labels)>limit or len(col_labels)>column_limit,
        "说明": "行列按频数降序；缺失单列为类别。归一化分母包含未展示类别，行列合计始终是原始计数。",
    }})


def linear_regression(dataset, x, y, filters=None):
    """用一元最小二乘拟合 y=截距+斜率*x，描述两个数值字段的样本关联。
    dataset 和 filters 确定范围，x 为自变量，y 为因变量，两者必须是不同字段。
    仅使用 x、y 同时为有限数的配对记录，返回 JSON 文本及有效、排除记录数。
    结果包含斜率、截距、R平方和样本内均方根误差，不代表样本外预测能力。
    有效配对不足2条或 x 为常量时不拟合；y 为常量时 R平方为 null。
    拟合不证明因果关系，结果需要结合字段含义与数据范围解释。
    """
    if x == y:
        raise ValueError("回归需要两个不同的数值字段")
    df, scope = _frame(dataset, filters)
    pair = pd.DataFrame({"x": _numeric(df, x), "y": _numeric(df, y)}).dropna()
    result = {"自变量": x, "因变量": y, "有效配对记录数": len(pair),
              "排除记录数": len(df)-len(pair), "斜率": None, "截距": None,
              "R平方": None, "均方根误差": None,
              "说明": "一元最小二乘拟合 y=截距+斜率*x；描述样本关联，不证明因果或预测能力。"}
    if len(pair) < 2 or pair["x"].nunique() < 2:
        result["提示"] = "有效记录不足2条或自变量为常量，无法拟合。"
    else:
        left, right = pair["x"].to_numpy(dtype=float), pair["y"].to_numpy(dtype=float)
        dx, dy = left-left.mean(), right-right.mean()
        slope = np.dot(dx, dy) / np.dot(dx, dx)
        intercept = right.mean() - slope*left.mean()
        residual = dy-slope*dx
        squared_error, total = np.dot(residual, residual), np.dot(dy, dy)
        result.update({"斜率": slope, "截距": intercept,
                       "R平方": 1-squared_error/total if total > 0 else None,
                       "均方根误差": np.sqrt(squared_error/len(pair))})
        if total == 0:
            result["提示"] = "因变量为常量，R平方无定义。"
    return _json({**scope, "结果": result})


# 以下是工具声明共用的参数格式；各函数的实际参数检查仍在上方 Python 实现中。
_FILTERS = {
    "type": "array",
    "description": "所有条件同时满足，作用于完整数据。字符串按原值比较（日期字符串不做日期解析）；contains为字面包含，between含两端；不改原CSV。",
    "items": {"type": "object", "properties": {
        "column": {"type": "string"},
        "op": {"type": "string", "enum": ["eq", "ne", "gt", "ge", "lt", "le", "in", "not_in", "between", "contains", "is_null", "not_null"]},
        "value": {"description": "与字段同类型的值；in/not_in用数组；between用[下界,上界]；is_null/not_null无需value。",
                  "anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}, {"type": "null"},
                            {"type": "array", "items": {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}, {"type": "null"}]}}]},
    }, "required": ["column", "op"], "additionalProperties": False},
}
_COLUMN = {"type": "string", "description": "原始字段名，先用 view_data 查看"}
_LIMIT = {"type": "integer", "minimum": 1, "maximum": 100, "description": "返回数量上限，统计仍使用完整筛选数据；超限标注截断"}
_OFFSET = {"type": "integer", "minimum": 0, "description": "从0开始的偏移量，用于翻页"}
_PERIOD = {"type": "string", "enum": ["day", "week", "month", "hour", "weekday"]}


def _tool(function, description, properties=None, required=()):
    """为独立函数生成 AI 工具声明，复用 dataset 和 filters 的参数格式。

    properties 描述该工具的专属参数，required 指定其中哪些必须由模型提供。
    此函数只组织工具说明，实际调用由主程序按工具名调度。
    """
    return {"type": "function", "function": {
        "name": function.__name__, "description": description,
        "parameters": {"type": "object", "properties": {
            "dataset": {"type": "string", "enum": list(FILES), "description": "seoul首尔租赁天气数据；jersey泽西市骑行记录，两个城市分别分析"},
            **(properties or {}), "filters": _FILTERS,
        }, "required": ["dataset", *required], "additionalProperties": False},
    }}


# 交给模型选择的独立工具列表；description 解释用途，parameters 约束调用参数。
ANALYSIS_TOOLS = [
    _tool(view_data, "查看本地数据的全部字段、类型、缺失计数和一页记录。可筛选、选列、排序、翻页；开始分析先调用此工具。", {
        "columns": {"type": "array", "items": _COLUMN, "minItems": 1, "uniqueItems": True},
        "sort_by": _COLUMN, "ascending": {"type": "boolean"}, "offset": _OFFSET, "limit": _LIMIT,
    }),
    _tool(mean_value, "计算一个数值字段的平均值、有效数及缺失或非有限数，可选按类别分组。", {
        "column": _COLUMN, "group_by": _COLUMN, "limit": _LIMIT,
    }, ("column",)),
    _tool(extract_categories, "提取字段的不同类别及频数、比例，按数量排名，支持翻页；筛选后可比较子群。", {
        "column": _COLUMN, "offset": _OFFSET, "limit": _LIMIT,
    }, ("column",)),
    _tool(describe_data, "查看整体或指定字段的描述统计，包括计数、均值、标准差、分位数或类别摘要，排除非有限数。", {
        "column": _COLUMN,
    }),
    _tool(check_data_quality, "检查重复行、缺失比例、非有限数、常量字段、时间范围和日期解析失败数；不自动清洗。", {
        "date_column": _COLUMN,
    }),
    _tool(group_statistics, "对类别分组后计算单项指标。count为每组记录数，无需column；其余方法需要数值字段，std/var为样本统计。", {
        "group_by": _COLUMN, "column": _COLUMN,
        "method": {"type": "string", "enum": ["mean", "sum", "min", "max", "median", "count", "std", "var"]},
        "limit": _LIMIT,
    }, ("group_by",)),
    _tool(compare_groups, "同时比较各组的记录数、有效数、均值、中位数、标准差、最小最大值和总和；按记录数降序。", {
        "group_by": _COLUMN, "column": _COLUMN, "limit": _LIMIT,
    }, ("group_by",)),
    _tool(time_trend, "按日、周、月或小时/星期周期计算记录数或数值字段统计；时间升序，不补零。首尔日期自动加小时，泽西市默认开始时间。", {
        "column": _COLUMN, "date_column": _COLUMN, "period": _PERIOD, "limit": _LIMIT,
    }),
    _tool(value_distribution, "计算数值字段的直方图、均值、中位数、标准差和分位数。", {
        "column": _COLUMN, "bins": {"type": "integer", "minimum": 1, "maximum": 50},
    }, ("column",)),
    _tool(detect_outliers, "用1.5倍IQR识别数值字段的异常候选数量与比例，保留原数据；候选不等于数据错误。", {
        "column": _COLUMN,
    }, ("column",)),
    _tool(correlation_analysis, "计算两个数值字段的Pearson或Spearman相关系数及有效配对数；相关性不代表因果。", {
        "column": _COLUMN, "other_column": _COLUMN,
        "method": {"type": "string", "enum": ["pearson", "spearman"]},
    }, ("column", "other_column")),
    _tool(quantile_analysis, "计算自选分位点，如P90/P95/P99，以分析长尾与阈值。", {
        "column": _COLUMN,
        "quantiles": {"type": "array", "minItems": 1, "maxItems": 20,
                      "items": {"type": "number", "minimum": 0, "maximum": 1}},
    }, ("column",)),
    _tool(cross_tabulate, "对两个类别字段做交叉计数或全体/行/列占比分析，缺失保留为类别；行列按频数降序，比例分母包含未展示类别。", {
        "row": _COLUMN, "column": _COLUMN,
        "normalize": {"type": "string", "enum": ["none", "all", "row", "column"]},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        "column_limit": {"type": "integer", "minimum": 1, "maximum": 50},
    }, ("row", "column")),
    _tool(linear_regression, "对两个数值字段做一元线性拟合，返回斜率、截距、R平方和样本内RMSE；不证明因果或样本外预测能力。", {
        "x": _COLUMN, "y": _COLUMN,
    }, ("x", "y")),
]

# 执行映射：主程序收到工具名后查找对应函数，用 **args 传参并取得 JSON 文本结果。
TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {function.__name__: function for function in (
    view_data, mean_value, extract_categories, describe_data, check_data_quality,
    group_statistics, compare_groups, time_trend, value_distribution, detect_outliers,
    correlation_analysis, quantile_analysis, cross_tabulate, linear_regression,
)}
# Python旧调用和已有会话仍可使用单体入口；新模型只看到上面的独立工具。
TOOL_FUNCTIONS["analyze_data"] = analyze_data

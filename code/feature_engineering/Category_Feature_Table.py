import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# 1. 读取清洗后的数据
# =========================

csv_path = "data/processed/Clean_User_Behavior.csv"
df = pd.read_csv(csv_path)


# =========================
# 2. 时间字段处理
# =========================

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday   # 0=周一，6=周日


# =========================
# 3. 行为类型映射
# =========================

# 原始数据中 behavior_type 使用数字编码
# 1 = pv 浏览；2 = fav 收藏；3 = cart 加购；4 = buy 购买
behavior_map = {
    1: "pv",
    2: "fav",
    3: "cart",
    4: "buy"
}

df["behavior_type"] = df["behavior_type"].map(behavior_map)


# =========================
# 4. 类目基础行为次数统计
# =========================

behavior_cnt = (
    df.pivot_table(
        index="item_category",
        columns="behavior_type",
        values="user_id",
        aggfunc="count",
        fill_value=0
    )
    .reset_index()
)

for col in ["pv", "fav", "cart", "buy"]:
    if col not in behavior_cnt.columns:
        behavior_cnt[col] = 0

behavior_cnt = behavior_cnt.rename(columns={
    "pv": "pv_cnt",
    "fav": "fav_cnt",
    "cart": "cart_cnt",
    "buy": "buy_cnt"
})

behavior_cnt = behavior_cnt[
    ["item_category", "pv_cnt", "fav_cnt", "cart_cnt", "buy_cnt"]
]


# =========================
# 5. 类目基础规模特征
# =========================

category_basic = (
    df.groupby("item_category")
    .agg(
        total_behavior_cnt=("behavior_type", "count"),
        unique_user_cnt=("user_id", "nunique"),
        unique_item_cnt=("item_id", "nunique"),
        active_days=("date", "nunique"),
        first_behavior_time=("timestamp", "min"),
        last_behavior_time=("timestamp", "max"),
    )
    .reset_index()
)

category_feature = category_basic.merge(
    behavior_cnt,
    on="item_category",
    how="left"
)


# =========================
# 6. 类目行为深度特征
# =========================

category_feature["avg_behavior_per_user"] = (
    category_feature["total_behavior_cnt"] /
    category_feature["unique_user_cnt"]
)

category_feature["avg_behavior_per_item"] = (
    category_feature["total_behavior_cnt"] /
    category_feature["unique_item_cnt"]
)

category_feature["avg_daily_behavior_cnt"] = (
    category_feature["total_behavior_cnt"] /
    category_feature["active_days"]
)

daily_behavior = (
    df.groupby(["item_category", "date"])
    .size()
    .reset_index(name="daily_behavior_cnt")
)

daily_behavior_stats = (
    daily_behavior.groupby("item_category")
    .agg(
        max_daily_behavior_cnt=("daily_behavior_cnt", "max"),
        min_daily_behavior_cnt=("daily_behavior_cnt", "min")
    )
    .reset_index()
)

category_feature = category_feature.merge(
    daily_behavior_stats,
    on="item_category",
    how="left"
)


# =========================
# 7. 类目转化率特征
# =========================

category_feature["pv_buy_ratio"] = np.where(
    category_feature["pv_cnt"] > 0,
    category_feature["buy_cnt"] / category_feature["pv_cnt"],
    np.nan
)

category_feature["cart_buy_ratio"] = np.where(
    category_feature["cart_cnt"] > 0,
    category_feature["buy_cnt"] / category_feature["cart_cnt"],
    np.nan
)

category_feature["fav_buy_ratio"] = np.where(
    category_feature["fav_cnt"] > 0,
    category_feature["buy_cnt"] / category_feature["fav_cnt"],
    np.nan
)

category_feature["pv_cart_ratio"] = np.where(
    category_feature["pv_cnt"] > 0,
    category_feature["cart_cnt"] / category_feature["pv_cnt"],
    np.nan
)


# =========================
# 8. 类目用户结构特征
# =========================

unique_view_user = (
    df[df["behavior_type"] == "pv"]
    .groupby("item_category")["user_id"]
    .nunique()
    .reset_index(name="unique_view_user_cnt")
)

unique_buy_user = (
    df[df["behavior_type"] == "buy"]
    .groupby("item_category")["user_id"]
    .nunique()
    .reset_index(name="unique_buy_user_cnt")
)

category_feature = category_feature.merge(
    unique_view_user,
    on="item_category",
    how="left"
)

category_feature = category_feature.merge(
    unique_buy_user,
    on="item_category",
    how="left"
)

category_feature[["unique_view_user_cnt", "unique_buy_user_cnt"]] = (
    category_feature[["unique_view_user_cnt", "unique_buy_user_cnt"]]
    .fillna(0)
    .astype(int)
)

category_feature["buy_user_ratio"] = np.where(
    category_feature["unique_view_user_cnt"] > 0,
    category_feature["unique_buy_user_cnt"] /
    category_feature["unique_view_user_cnt"],
    np.nan
)


# =========================
# 9. 类目时间活跃特征
# =========================

category_feature["category_lifetime_days"] = (
    category_feature["last_behavior_time"].dt.normalize()
    - category_feature["first_behavior_time"].dt.normalize()
).dt.days + 1


# 类目最热小时
hour_behavior = (
    df.groupby(["item_category", "hour"])
    .size()
    .reset_index(name="hour_behavior_cnt")
)

peak_hour = (
    hour_behavior.sort_values(
        ["item_category", "hour_behavior_cnt"],
        ascending=[True, False]
    )
    .drop_duplicates("item_category")
    [["item_category", "hour"]]
    .rename(columns={"hour": "peak_active_hour"})
)

category_feature = category_feature.merge(
    peak_hour,
    on="item_category",
    how="left"
)


# 周末行为占比
weekend_behavior = (
    df[df["weekday"] >= 5]
    .groupby("item_category")
    .size()
    .reset_index(name="weekend_behavior_cnt")
)

category_feature = category_feature.merge(
    weekend_behavior,
    on="item_category",
    how="left"
)

category_feature["weekend_behavior_cnt"] = (
    category_feature["weekend_behavior_cnt"]
    .fillna(0)
)

category_feature["weekend_behavior_ratio"] = (
    category_feature["weekend_behavior_cnt"] /
    category_feature["total_behavior_cnt"]
)


# 深夜行为占比：0:00–6:00
late_night_behavior = (
    df[(df["hour"] >= 0) & (df["hour"] < 6)]
    .groupby("item_category")
    .size()
    .reset_index(name="late_night_behavior_cnt")
)

category_feature = category_feature.merge(
    late_night_behavior,
    on="item_category",
    how="left"
)

category_feature["late_night_behavior_cnt"] = (
    category_feature["late_night_behavior_cnt"]
    .fillna(0)
)

category_feature["late_night_behavior_ratio"] = (
    category_feature["late_night_behavior_cnt"] /
    category_feature["total_behavior_cnt"]
)


# =========================
# 10. 类目集中度特征
# =========================

# 类目下每个商品的行为次数
category_item_behavior = (
    df.groupby(["item_category", "item_id"])
    .size()
    .reset_index(name="item_behavior_cnt")
)

# 每个类目下最热门商品的行为次数
top_item_behavior = (
    category_item_behavior.groupby("item_category")
    .agg(
        top_item_behavior_cnt=("item_behavior_cnt", "max")
    )
    .reset_index()
)

category_feature = category_feature.merge(
    top_item_behavior,
    on="item_category",
    how="left"
)

category_feature["top_item_ratio"] = (
    category_feature["top_item_behavior_cnt"] /
    category_feature["total_behavior_cnt"]
)


# =========================
# 11. 数据格式优化
# =========================

float_cols = category_feature.select_dtypes(include="float").columns
category_feature[float_cols] = category_feature[float_cols].round(4)


# =========================
# 12. 数据检查
# =========================

print("\n类目中间表字段信息：")
print(category_feature.info())

print("\n字段列表：")
print(category_feature.columns.tolist())

print("\n缺失值统计：")
print(category_feature.isnull().sum())

print("\n数值字段统计：")
print(category_feature.describe())


# =========================
# 13. 输出类目中间表
# =========================

output_path = Path("data/intermediate_tables/Category_Feature_Table.parquet")

category_feature.to_parquet(
    output_path,
    index=False,
    engine="pyarrow"
)

print(f"\n类目中间表已保存至：{output_path}")
print(category_feature.head())
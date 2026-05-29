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
# 4. 商品基础行为次数统计
# =========================

behavior_cnt = (
    df.pivot_table(
        index="item_id",
        columns="behavior_type",
        values="user_id",
        aggfunc="count",
        fill_value=0
    )
    .reset_index()
)

# 防止某些行为类型不存在
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
    ["item_id", "pv_cnt", "fav_cnt", "cart_cnt", "buy_cnt"]
]


# =========================
# 5. 商品基础规模特征
# =========================

item_basic = (
    df.groupby("item_id")
    .agg(
        total_behavior_cnt=("behavior_type", "count"),
        unique_user_cnt=("user_id", "nunique"),
        active_days=("date", "nunique"),
        first_behavior_time=("timestamp", "min"),
        last_behavior_time=("timestamp", "max"),
    )
    .reset_index()
)

item_feature = item_basic.merge(behavior_cnt, on="item_id", how="left")


# =========================
# 6. 商品行为深度特征
# =========================

item_feature["avg_behavior_per_user"] = (
    item_feature["total_behavior_cnt"] / item_feature["unique_user_cnt"]
)

item_feature["avg_daily_behavior_cnt"] = (
    item_feature["total_behavior_cnt"] / item_feature["active_days"]
)

daily_behavior = (
    df.groupby(["item_id", "date"])
    .size()
    .reset_index(name="daily_behavior_cnt")
)

daily_behavior_stats = (
    daily_behavior.groupby("item_id")
    .agg(
        max_daily_behavior_cnt=("daily_behavior_cnt", "max"),
        min_daily_behavior_cnt=("daily_behavior_cnt", "min")
    )
    .reset_index()
)

item_feature = item_feature.merge(
    daily_behavior_stats,
    on="item_id",
    how="left"
)


# =========================
# 7. 商品转化率特征
# =========================

item_feature["pv_buy_ratio"] = np.where(
    item_feature["pv_cnt"] > 0,
    item_feature["buy_cnt"] / item_feature["pv_cnt"],
    np.nan
)

item_feature["cart_buy_ratio"] = np.where(
    item_feature["cart_cnt"] > 0,
    item_feature["buy_cnt"] / item_feature["cart_cnt"],
    np.nan
)

item_feature["fav_buy_ratio"] = np.where(
    item_feature["fav_cnt"] > 0,
    item_feature["buy_cnt"] / item_feature["fav_cnt"],
    np.nan
)

item_feature["pv_cart_ratio"] = np.where(
    item_feature["pv_cnt"] > 0,
    item_feature["cart_cnt"] / item_feature["pv_cnt"],
    np.nan
)


# =========================
# 8. 商品用户结构特征
# =========================

unique_view_user = (
    df[df["behavior_type"] == "pv"]
    .groupby("item_id")["user_id"]
    .nunique()
    .reset_index(name="unique_view_user_cnt")
)

unique_buy_user = (
    df[df["behavior_type"] == "buy"]
    .groupby("item_id")["user_id"]
    .nunique()
    .reset_index(name="unique_buy_user_cnt")
)

item_feature = item_feature.merge(
    unique_view_user,
    on="item_id",
    how="left"
)

item_feature = item_feature.merge(
    unique_buy_user,
    on="item_id",
    how="left"
)

item_feature[["unique_view_user_cnt", "unique_buy_user_cnt"]] = (
    item_feature[["unique_view_user_cnt", "unique_buy_user_cnt"]]
    .fillna(0)
    .astype(int)
)

item_feature["buy_user_ratio"] = np.where(
    item_feature["unique_view_user_cnt"] > 0,
    item_feature["unique_buy_user_cnt"] / item_feature["unique_view_user_cnt"],
    np.nan
)


# =========================
# 9. 商品时间活跃特征
# =========================

item_feature["item_lifetime_days"] = (
    item_feature["last_behavior_time"].dt.normalize()
    - item_feature["first_behavior_time"].dt.normalize()
).dt.days + 1


# 商品最热小时
hour_behavior = (
    df.groupby(["item_id", "hour"])
    .size()
    .reset_index(name="hour_behavior_cnt")
)

peak_hour = (
    hour_behavior.sort_values(
        ["item_id", "hour_behavior_cnt"],
        ascending=[True, False]
    )
    .drop_duplicates("item_id")
    [["item_id", "hour"]]
    .rename(columns={"hour": "peak_active_hour"})
)

item_feature = item_feature.merge(
    peak_hour,
    on="item_id",
    how="left"
)


# 周末行为占比
weekend_behavior = (
    df[df["weekday"] >= 5]
    .groupby("item_id")
    .size()
    .reset_index(name="weekend_behavior_cnt")
)

item_feature = item_feature.merge(
    weekend_behavior,
    on="item_id",
    how="left"
)

item_feature["weekend_behavior_cnt"] = (
    item_feature["weekend_behavior_cnt"]
    .fillna(0)
)

item_feature["weekend_behavior_ratio"] = (
    item_feature["weekend_behavior_cnt"] /
    item_feature["total_behavior_cnt"]
)


# 深夜行为占比：0:00–6:00
late_night_behavior = (
    df[(df["hour"] >= 0) & (df["hour"] < 6)]
    .groupby("item_id")
    .size()
    .reset_index(name="late_night_behavior_cnt")
)

item_feature = item_feature.merge(
    late_night_behavior,
    on="item_id",
    how="left"
)

item_feature["late_night_behavior_cnt"] = (
    item_feature["late_night_behavior_cnt"]
    .fillna(0)
)

item_feature["late_night_behavior_ratio"] = (
    item_feature["late_night_behavior_cnt"] /
    item_feature["total_behavior_cnt"]
)


# =========================
# 10. 数据格式优化
# =========================

# float字段保留4位小数
float_cols = item_feature.select_dtypes(include="float").columns
item_feature[float_cols] = item_feature[float_cols].round(4)


# =========================
# 11. 数据检查
# =========================

print("\n商品中间表字段信息：")
print(item_feature.info())

print("\n字段列表：")
print(item_feature.columns.tolist())

print("\n缺失值统计：")
print(item_feature.isnull().sum())

print("\n数值字段统计：")
print(item_feature.describe())


# =========================
# 12. 输出商品中间表
# =========================

output_path = Path("data/intermediate_tables/Item_Feature_Table.parquet")

item_feature.to_parquet(
    output_path,
    index=False,
    engine="pyarrow"
)

print(f"\n商品中间表已保存至：{output_path}")
print(item_feature.head())
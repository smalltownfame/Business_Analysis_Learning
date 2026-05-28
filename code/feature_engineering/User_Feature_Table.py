import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 读取原始数据
# =========================
csv_path = "data/processed/Clean_User_Behavior.csv"
df = pd.read_csv(csv_path)


# =========================
# 时间字段处理
# =========================
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday   # 0=周一，6=周日


# =========================
# 基础行为次数统计
# =========================
behavior_map = {1: "pv", 2: "cart", 3: "fav", 4: "buy"} # 行为类型映射：原始数据中 1/2/3/4 分别代表 pv/cart/fav/buy
df["behavior_type"] = df["behavior_type"].map(behavior_map)

behavior_cnt = (
    df.pivot_table(
        index="user_id",
        columns="behavior_type",
        values="item_id",
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

# 保留需要的列，避免多余列进入中间表
behavior_cnt = behavior_cnt[
    ["user_id", "pv_cnt", "fav_cnt", "cart_cnt", "buy_cnt"]
]


# =========================
# 用户总体行为特征
# =========================
user_basic = df.groupby("user_id").agg(
    total_behavior_cnt=("behavior_type", "count"),
    active_days=("date", "nunique"),
    behavior_type_cnt=("behavior_type", "nunique"),
    first_behavior_time=("timestamp", "min"),
    last_behavior_time=("timestamp", "max"),
    active_hour_cnt=("hour", "nunique")
).reset_index()


# =========================
# 合并基础行为表
# =========================
user_feature = user_basic.merge(behavior_cnt, on="user_id", how="left")


# =========================
# 行为深度特征
# =========================
user_feature["avg_behavior_per_day"] = (
    user_feature["total_behavior_cnt"] / user_feature["active_days"]
)


# =========================
# 转化率特征
# =========================
user_feature["user_buy_rate"] = (
    user_feature["buy_cnt"] / user_feature["total_behavior_cnt"]
)

user_feature["user_cart_rate"] = (
    user_feature["cart_cnt"] / user_feature["total_behavior_cnt"]
)

user_feature["user_fav_rate"] = (
    user_feature["fav_cnt"] / user_feature["total_behavior_cnt"]
)

user_feature["user_pv_rate"] = (
    user_feature["pv_cnt"] / user_feature["total_behavior_cnt"]
)

# 避免除以 0
user_feature["user_intent_score"] = np.where(
    user_feature["pv_cnt"] > 0,
    (user_feature["buy_cnt"] + user_feature["cart_cnt"]) / user_feature["pv_cnt"],
    np.nan
)

user_feature["cart_buy_ratio"] = np.where(
    user_feature["cart_cnt"] > 0,
    user_feature["buy_cnt"] / user_feature["cart_cnt"],
    np.nan
)

user_feature["fav_buy_ratio"] = np.where(
    user_feature["fav_cnt"] > 0,
    user_feature["buy_cnt"] / user_feature["fav_cnt"],
    np.nan
)

user_feature["pv_buy_ratio"] = np.where(
    user_feature["pv_cnt"] > 0,
    user_feature["buy_cnt"] / user_feature["pv_cnt"],
    np.nan
)


# =========================
# 用户行为时间跨度
# =========================
user_feature["first_behavior_time"] = pd.to_datetime(user_feature["first_behavior_time"]) # 确保时间字段是 datetime 类型
user_feature["last_behavior_time"] = pd.to_datetime(user_feature["last_behavior_time"]) # 确保时间字段是 datetime 类型

# 用户行为时间跨度
user_feature["user_lifetime_days"] = (
    user_feature["last_behavior_time"].dt.normalize()
    - user_feature["first_behavior_time"].dt.normalize()
).dt.days + 1


# =========================
# 时间段行为特征
# =========================
def count_time_range(data, start_hour, end_hour):
    return data[(data["hour"] >= start_hour) & (data["hour"] < end_hour)] \
        .groupby("user_id") \
        .size()

time_features = pd.DataFrame({"user_id": user_feature["user_id"]})

for name, start, end in [
    ("late_night_behavior_cnt", 0, 6),
    ("morning_behavior_cnt", 6, 12),
    ("afternoon_behavior_cnt", 12, 18),
    ("evening_behavior_cnt", 18, 24),
]:
    temp = count_time_range(df, start, end).reset_index(name=name)
    time_features = time_features.merge(temp, on="user_id", how="left")

time_features = time_features.fillna(0)

user_feature = user_feature.merge(time_features, on="user_id", how="left")

user_feature["morning_ratio"] = user_feature["morning_behavior_cnt"] / user_feature["total_behavior_cnt"]
user_feature["evening_ratio"] = user_feature["evening_behavior_cnt"] / user_feature["total_behavior_cnt"]
user_feature["late_night_ratio"] = user_feature["late_night_behavior_cnt"] / user_feature["total_behavior_cnt"]


# =========================
# 商品兴趣特征
# =========================
item_features = df.groupby("user_id").agg(
    item_diversity=("item_id", "nunique"),
    category_diversity=("item_category", "nunique")
).reset_index()

user_feature = user_feature.merge(item_features, on="user_id", how="left")

user_feature["avg_behavior_per_item"] = (
    user_feature["total_behavior_cnt"] / user_feature["item_diversity"]
)

user_feature["avg_behavior_per_category"] = (
    user_feature["total_behavior_cnt"] / user_feature["category_diversity"]
)


# =========================
# 用户价值标签
# =========================
user_feature["repurchase_flag"] = user_feature["buy_cnt"] > 1

user_feature["active_user_flag"] = (
    user_feature["active_days"] >= user_feature["active_days"].quantile(0.75)
)

user_feature["high_value_user_flag"] = (
    user_feature["buy_cnt"] >= user_feature["buy_cnt"].quantile(0.75)
)

user_feature["potential_user_flag"] = (
    (user_feature["cart_cnt"] + user_feature["fav_cnt"] >= 
     (user_feature["cart_cnt"] + user_feature["fav_cnt"]).quantile(0.75))
    & (user_feature["buy_cnt"] <= user_feature["buy_cnt"].median())
)


# =========================
# 数据检查
# =========================
print("\n用户中间表字段信息：")
print(user_feature.info())

print("\n字段列表：")
print(user_feature.columns.tolist())

print("\n缺失值统计：")
print(user_feature.isnull().sum())

print("\n数值字段统计：")
print(user_feature.describe())


# =========================
# 数据格式优化
# =========================
float_cols = user_feature.select_dtypes(include="float").columns
user_feature[float_cols] = user_feature[float_cols].round(4) # float字段保留4位小数


# =========================
# 输出中间表
# =========================
user_feature_path = Path("data/processed/User_Feature_Table.csv") # 用户中间表保存路径

# 保存 CSV
user_feature.to_csv(
    user_feature_path,
    index=False,
    encoding="utf-8-sig"
)
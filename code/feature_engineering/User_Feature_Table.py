import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# 读取清洗后的行为明细数据
# =========================

csv_path = "data/processed/Clean_User_Behavior.csv"
df = pd.read_csv(csv_path)


# =========================
# 1. 基础预处理
# =========================

# 时间字段转换为 datetime，便于提取日期、小时和星期
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
df["date"] = pd.to_datetime(df["date"])
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday   # 0=周一，6=周日


# 行为类型映射：
# 原始数据中 behavior_type 使用数字编码
# 1 = pv 浏览；2 = cart 加购；3 = fav 收藏；4 = buy 购买
behavior_map = {
    1: "pv",
    2: "cart",
    3: "fav",
    4: "buy"
}

df["behavior_type"] = df["behavior_type"].map(behavior_map)

# 为了计算行为序列特征，先按用户和时间排序
df_sorted = df.sort_values(["user_id", "timestamp"]).copy()


# =========================
# 2. 基础行为规模特征
# =========================

# 统计每个用户四类行为次数：浏览、收藏、加购、购买
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

# 防止某类行为在样本中不存在，导致后续字段缺失
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
    ["user_id", "pv_cnt", "fav_cnt", "cart_cnt", "buy_cnt"]
]


# 用户总体行为统计：
# total_behavior_cnt：用户总行为次数
# active_days：用户活跃天数
# behavior_type_cnt：用户产生过的行为类型数量
# first_behavior_time / last_behavior_time：用户首次 / 最近一次行为时间
# active_hour_cnt：用户出现过行为的不同小时数量
user_basic = (
    df.groupby("user_id")
    .agg(
        total_behavior_cnt=("behavior_type", "count"),
        active_days=("date", "nunique"),
        behavior_type_cnt=("behavior_type", "nunique"),
        first_behavior_time=("timestamp", "min"),
        last_behavior_time=("timestamp", "max"),
        active_hour_cnt=("hour", "nunique")
    )
    .reset_index()
)

user_feature = user_basic.merge(
    behavior_cnt,
    on="user_id",
    how="left"
)


# =========================
# 3. 行为深度特征
# =========================

# avg_behavior_per_day：
# 用户日均行为次数 = 用户总行为次数 / 用户活跃天数
user_feature["avg_behavior_per_day"] = (
    user_feature["total_behavior_cnt"] / user_feature["active_days"]
)

# 统计用户每天的行为次数，用于计算单日行为峰值和最低值
daily_behavior = (
    df.groupby(["user_id", "date"])
    .size()
    .reset_index(name="daily_behavior_cnt")
)

daily_behavior_stats = (
    daily_behavior.groupby("user_id")
    .agg(
        max_daily_behavior_cnt=("daily_behavior_cnt", "max"),
        min_daily_behavior_cnt=("daily_behavior_cnt", "min")
    )
    .reset_index()
)

user_feature = user_feature.merge(
    daily_behavior_stats,
    on="user_id",
    how="left"
)


# =========================
# 4. 转化率特征
# =========================

# user_buy_rate：
# 购买行为占比 = 购买次数 / 总行为次数
user_feature["user_buy_rate"] = (
    user_feature["buy_cnt"] / user_feature["total_behavior_cnt"]
)

# user_cart_rate：
# 加购行为占比 = 加购次数 / 总行为次数
user_feature["user_cart_rate"] = (
    user_feature["cart_cnt"] / user_feature["total_behavior_cnt"]
)

# user_fav_rate：
# 收藏行为占比 = 收藏次数 / 总行为次数
user_feature["user_fav_rate"] = (
    user_feature["fav_cnt"] / user_feature["total_behavior_cnt"]
)

# user_pv_rate：
# 浏览行为占比 = 浏览次数 / 总行为次数
user_feature["user_pv_rate"] = (
    user_feature["pv_cnt"] / user_feature["total_behavior_cnt"]
)

# user_intent_score：
# 用户购买倾向系数 = (购买次数 + 加购次数) / 浏览次数
# 分母为0时保留 NaN，表示该用户没有浏览行为，无法计算倾向系数
user_feature["user_intent_score"] = np.where(
    user_feature["pv_cnt"] > 0,
    (user_feature["buy_cnt"] + user_feature["cart_cnt"]) / user_feature["pv_cnt"],
    np.nan
)

# cart_buy_ratio：
# 加购到购买转化能力 = 购买次数 / 加购次数
user_feature["cart_buy_ratio"] = np.where(
    user_feature["cart_cnt"] > 0,
    user_feature["buy_cnt"] / user_feature["cart_cnt"],
    np.nan
)

# fav_buy_ratio：
# 收藏到购买转化能力 = 购买次数 / 收藏次数
user_feature["fav_buy_ratio"] = np.where(
    user_feature["fav_cnt"] > 0,
    user_feature["buy_cnt"] / user_feature["fav_cnt"],
    np.nan
)

# pv_buy_ratio：
# 浏览到购买转化能力 = 购买次数 / 浏览次数
user_feature["pv_buy_ratio"] = np.where(
    user_feature["pv_cnt"] > 0,
    user_feature["buy_cnt"] / user_feature["pv_cnt"],
    np.nan
)

# pv_to_cart_ratio：
# 浏览后加购比例 = 加购次数 / 浏览次数
user_feature["pv_to_cart_ratio"] = np.where(
    user_feature["pv_cnt"] > 0,
    user_feature["cart_cnt"] / user_feature["pv_cnt"],
    np.nan
)

# hesitation_score：
# 用户下单犹豫度 = 浏览次数 / 购买次数
# 仅对已购买用户计算，buy_cnt = 0 时保留 NaN
user_feature["hesitation_score"] = np.where(
    user_feature["buy_cnt"] > 0,
    user_feature["pv_cnt"] / user_feature["buy_cnt"],
    np.nan
)


# =========================
# 5. 用户行为时间跨度
# =========================

user_feature["first_behavior_time"] = pd.to_datetime(user_feature["first_behavior_time"])
user_feature["last_behavior_time"] = pd.to_datetime(user_feature["last_behavior_time"])

# user_lifetime_days：
# 用户行为时间跨度 = 最近一次行为日期 - 首次行为日期 + 1
user_feature["user_lifetime_days"] = (
    user_feature["last_behavior_time"].dt.normalize()
    - user_feature["first_behavior_time"].dt.normalize()
).dt.days + 1


# =========================
# 6. 时间活跃特征
# =========================

def count_time_range(data, start_hour, end_hour):
    """统计指定小时区间内每个用户的行为次数。"""
    return (
        data[(data["hour"] >= start_hour) & (data["hour"] < end_hour)]
        .groupby("user_id")
        .size()
    )


# 统计凌晨、上午、下午、晚间四个时间段行为次数
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

user_feature = user_feature.merge(
    time_features,
    on="user_id",
    how="left"
)

# morning_ratio：
# 上午行为占比 = 上午行为次数 / 总行为次数
user_feature["morning_ratio"] = (
    user_feature["morning_behavior_cnt"] / user_feature["total_behavior_cnt"]
)

# evening_ratio：
# 晚间行为占比 = 晚间行为次数 / 总行为次数
user_feature["evening_ratio"] = (
    user_feature["evening_behavior_cnt"] / user_feature["total_behavior_cnt"]
)

# late_night_ratio：
# 深夜行为占比 = 深夜行为次数 / 总行为次数
user_feature["late_night_ratio"] = (
    user_feature["late_night_behavior_cnt"] / user_feature["total_behavior_cnt"]
)


# weekend_behavior_cnt：
# 周末行为次数，weekday >= 5 表示周六或周日
weekend_behavior = (
    df[df["weekday"] >= 5]
    .groupby("user_id")
    .size()
    .reset_index(name="weekend_behavior_cnt")
)

# weekday_behavior_cnt：
# 工作日行为次数，weekday < 5 表示周一至周五
weekday_behavior = (
    df[df["weekday"] < 5]
    .groupby("user_id")
    .size()
    .reset_index(name="weekday_behavior_cnt")
)

user_feature = user_feature.merge(weekend_behavior, on="user_id", how="left")
user_feature = user_feature.merge(weekday_behavior, on="user_id", how="left")

user_feature[["weekend_behavior_cnt", "weekday_behavior_cnt"]] = (
    user_feature[["weekend_behavior_cnt", "weekday_behavior_cnt"]]
    .fillna(0)
    .astype(int)
)

# weekend_ratio：
# 周末行为占比 = 周末行为次数 / 总行为次数
user_feature["weekend_ratio"] = (
    user_feature["weekend_behavior_cnt"] / user_feature["total_behavior_cnt"]
)


# peak_active_hour：
# 用户最活跃小时 = 行为次数最多的小时
hour_behavior = (
    df.groupby(["user_id", "hour"])
    .size()
    .reset_index(name="hour_behavior_cnt")
)

peak_hour = (
    hour_behavior
    .sort_values(["user_id", "hour_behavior_cnt"], ascending=[True, False])
    .drop_duplicates("user_id")
    [["user_id", "hour"]]
    .rename(columns={"hour": "peak_active_hour"})
)

user_feature = user_feature.merge(
    peak_hour,
    on="user_id",
    how="left"
)


# =========================
# 7. 时间粒度特征
# =========================

# active_1h_slot：
# 用户活跃过的1小时时段数量，本质上等同于 active_hour_cnt
user_feature["active_1h_slot"] = user_feature["active_hour_cnt"]

# 创建2小时、3小时、6小时时段编号
df["slot_2h"] = (df["hour"] // 2).astype(int)
df["slot_3h"] = (df["hour"] // 3).astype(int)
df["slot_6h"] = (df["hour"] // 6).astype(int)

# active_2h_slot：
# 用户活跃过的2小时时段数量
active_2h_slot = (
    df.groupby("user_id")["slot_2h"]
    .nunique()
    .reset_index(name="active_2h_slot")
)

# active_3h_slot：
# 用户活跃过的3小时时段数量
active_3h_slot = (
    df.groupby("user_id")["slot_3h"]
    .nunique()
    .reset_index(name="active_3h_slot")
)

# active_6h_slot：
# 用户活跃过的6小时时段数量
active_6h_slot = (
    df.groupby("user_id")["slot_6h"]
    .nunique()
    .reset_index(name="active_6h_slot")
)

for temp in [active_2h_slot, active_3h_slot, active_6h_slot]:
    user_feature = user_feature.merge(temp, on="user_id", how="left")

# peak_1h_slot：
# 行为最多的1小时时段；与 peak_active_hour 一致，为了与特征体系保持一致保留字段
user_feature["peak_1h_slot"] = user_feature["peak_active_hour"]

# peak_6h_slot：
# 行为最多的6小时时段
slot_6h_behavior = (
    df.groupby(["user_id", "slot_6h"])
    .size()
    .reset_index(name="slot_6h_behavior_cnt")
)

peak_6h_slot = (
    slot_6h_behavior
    .sort_values(["user_id", "slot_6h_behavior_cnt"], ascending=[True, False])
    .drop_duplicates("user_id")
    [["user_id", "slot_6h"]]
    .rename(columns={"slot_6h": "peak_6h_slot"})
)

user_feature = user_feature.merge(
    peak_6h_slot,
    on="user_id",
    how="left"
)


# =========================
# 8. 商品兴趣与商品偏好特征
# =========================

# item_diversity：
# 用户产生过行为的不同商品数量，衡量商品兴趣广度
# category_diversity：
# 用户产生过行为的不同类目数量，衡量类目兴趣广度
item_features = (
    df.groupby("user_id")
    .agg(
        item_diversity=("item_id", "nunique"),
        category_diversity=("item_category", "nunique")
    )
    .reset_index()
)

user_feature = user_feature.merge(
    item_features,
    on="user_id",
    how="left"
)

# avg_behavior_per_item：
# 单商品平均行为次数 = 总行为次数 / 不同商品数量
user_feature["avg_behavior_per_item"] = (
    user_feature["total_behavior_cnt"] / user_feature["item_diversity"]
)

# avg_behavior_per_category：
# 单类目平均行为次数 = 总行为次数 / 不同类目数量
user_feature["avg_behavior_per_category"] = (
    user_feature["total_behavior_cnt"] / user_feature["category_diversity"]
)


# viewed_item_cnt：
# 用户浏览过的不同商品数量
viewed_item_cnt = (
    df[df["behavior_type"] == "pv"]
    .groupby("user_id")["item_id"]
    .nunique()
    .reset_index(name="viewed_item_cnt")
)

# bought_item_cnt：
# 用户购买过的不同商品数量
bought_item_cnt = (
    df[df["behavior_type"] == "buy"]
    .groupby("user_id")["item_id"]
    .nunique()
    .reset_index(name="bought_item_cnt")
)

# cart_item_cnt：
# 用户加购过的不同商品数量
cart_item_cnt = (
    df[df["behavior_type"] == "cart"]
    .groupby("user_id")["item_id"]
    .nunique()
    .reset_index(name="cart_item_cnt")
)

# fav_item_cnt：
# 用户收藏过的不同商品数量
fav_item_cnt = (
    df[df["behavior_type"] == "fav"]
    .groupby("user_id")["item_id"]
    .nunique()
    .reset_index(name="fav_item_cnt")
)

for temp in [viewed_item_cnt, bought_item_cnt, cart_item_cnt, fav_item_cnt]:
    user_feature = user_feature.merge(temp, on="user_id", how="left")

item_count_cols = [
    "viewed_item_cnt",
    "bought_item_cnt",
    "cart_item_cnt",
    "fav_item_cnt"
]

user_feature[item_count_cols] = (
    user_feature[item_count_cols]
    .fillna(0)
    .astype(int)
)


# repeat_view_item_cnt：
# 浏览次数 > 1 的商品数量，衡量用户是否对商品反复关注
view_item_behavior = (
    df[df["behavior_type"] == "pv"]
    .groupby(["user_id", "item_id"])
    .size()
    .reset_index(name="view_cnt")
)

repeat_view_item_cnt = (
    view_item_behavior[view_item_behavior["view_cnt"] > 1]
    .groupby("user_id")
    .size()
    .reset_index(name="repeat_view_item_cnt")
)

# repeat_buy_item_cnt：
# 购买次数 > 1 的商品数量，衡量用户是否重复购买同一商品
buy_item_behavior = (
    df[df["behavior_type"] == "buy"]
    .groupby(["user_id", "item_id"])
    .size()
    .reset_index(name="buy_cnt_item")
)

repeat_buy_item_cnt = (
    buy_item_behavior[buy_item_behavior["buy_cnt_item"] > 1]
    .groupby("user_id")
    .size()
    .reset_index(name="repeat_buy_item_cnt")
)

user_feature = user_feature.merge(
    repeat_view_item_cnt,
    on="user_id",
    how="left"
)

user_feature = user_feature.merge(
    repeat_buy_item_cnt,
    on="user_id",
    how="left"
)

user_feature[["repeat_view_item_cnt", "repeat_buy_item_cnt"]] = (
    user_feature[["repeat_view_item_cnt", "repeat_buy_item_cnt"]]
    .fillna(0)
    .astype(int)
)


# top_item_id：
# 用户行为次数最多的商品，代表用户最感兴趣商品
# top_item_behavior_cnt：
# 用户在最偏好商品上的行为次数
user_item_behavior = (
    df.groupby(["user_id", "item_id"])
    .size()
    .reset_index(name="item_behavior_cnt")
)

top_item = (
    user_item_behavior
    .sort_values(["user_id", "item_behavior_cnt"], ascending=[True, False])
    .drop_duplicates("user_id")
    .rename(columns={
        "item_id": "top_item_id",
        "item_behavior_cnt": "top_item_behavior_cnt"
    })
)

user_feature = user_feature.merge(
    top_item[["user_id", "top_item_id", "top_item_behavior_cnt"]],
    on="user_id",
    how="left"
)

# favorite_item_ratio：
# 商品偏好集中度 = 最偏好商品行为次数 / 总行为次数
user_feature["favorite_item_ratio"] = (
    user_feature["top_item_behavior_cnt"] / user_feature["total_behavior_cnt"]
)

# repeat_buy_same_item_flag：
# 是否重复购买同一商品，判断标准：repeat_buy_item_cnt > 0
user_feature["repeat_buy_same_item_flag"] = (
    user_feature["repeat_buy_item_cnt"] > 0
)

# single_item_dependency_flag：
# 单商品依赖用户，判断标准：商品偏好集中度高于75分位数
user_feature["single_item_dependency_flag"] = (
    user_feature["favorite_item_ratio"] >=
    user_feature["favorite_item_ratio"].quantile(0.75)
)


# =========================
# 9. 类目兴趣与类目偏好特征
# =========================

# viewed_category_cnt：
# 用户浏览过的不同类目数量
viewed_category_cnt = (
    df[df["behavior_type"] == "pv"]
    .groupby("user_id")["item_category"]
    .nunique()
    .reset_index(name="viewed_category_cnt")
)

# bought_category_cnt：
# 用户购买过的不同类目数量
bought_category_cnt = (
    df[df["behavior_type"] == "buy"]
    .groupby("user_id")["item_category"]
    .nunique()
    .reset_index(name="bought_category_cnt")
)

# cart_category_cnt：
# 用户加购过的不同类目数量
cart_category_cnt = (
    df[df["behavior_type"] == "cart"]
    .groupby("user_id")["item_category"]
    .nunique()
    .reset_index(name="cart_category_cnt")
)

# fav_category_cnt：
# 用户收藏过的不同类目数量
fav_category_cnt = (
    df[df["behavior_type"] == "fav"]
    .groupby("user_id")["item_category"]
    .nunique()
    .reset_index(name="fav_category_cnt")
)

for temp in [
    viewed_category_cnt,
    bought_category_cnt,
    cart_category_cnt,
    fav_category_cnt
]:
    user_feature = user_feature.merge(temp, on="user_id", how="left")

category_count_cols = [
    "viewed_category_cnt",
    "bought_category_cnt",
    "cart_category_cnt",
    "fav_category_cnt"
]

user_feature[category_count_cols] = (
    user_feature[category_count_cols]
    .fillna(0)
    .astype(int)
)


# top_category_id：
# 用户行为次数最多的类目，代表用户最偏好类目
# top_category_behavior_cnt：
# 用户在最偏好类目上的行为次数
user_category_behavior = (
    df.groupby(["user_id", "item_category"])
    .size()
    .reset_index(name="category_behavior_cnt")
)

top_category = (
    user_category_behavior
    .sort_values(["user_id", "category_behavior_cnt"], ascending=[True, False])
    .drop_duplicates("user_id")
    .rename(columns={
        "item_category": "top_category_id",
        "category_behavior_cnt": "top_category_behavior_cnt"
    })
)

user_feature = user_feature.merge(
    top_category[["user_id", "top_category_id", "top_category_behavior_cnt"]],
    on="user_id",
    how="left"
)

# top_category_behavior_ratio：
# 类目偏好集中度 = 最偏好类目行为次数 / 总行为次数
user_feature["top_category_behavior_ratio"] = (
    user_feature["top_category_behavior_cnt"] /
    user_feature["total_behavior_cnt"]
)

# cross_category_user_flag：
# 是否为跨类目用户，判断标准：类目兴趣广度高于75分位数
user_feature["cross_category_user_flag"] = (
    user_feature["category_diversity"] >=
    user_feature["category_diversity"].quantile(0.75)
)


# =========================
# 10. 行为序列特征
# =========================

# behavior_order：
# 用户行为序号，用于计算购买前行为步数
df_sorted["behavior_order"] = (
    df_sorted.groupby("user_id")
    .cumcount() + 1
)

# avg_steps_to_buy：
# 每次购买前平均行为次数
# 计算逻辑：对每次购买行为，取该购买行为在用户行为序列中的位置 - 1，然后求均值
buy_steps = df_sorted[df_sorted["behavior_type"] == "buy"][
    ["user_id", "behavior_order"]
].copy()

buy_steps["steps_before_buy"] = buy_steps["behavior_order"] - 1

avg_steps_to_buy = (
    buy_steps.groupby("user_id")["steps_before_buy"]
    .mean()
    .reset_index(name="avg_steps_to_buy")
)

user_feature = user_feature.merge(
    avg_steps_to_buy,
    on="user_id",
    how="left"
)


# first_buy_interval：
# 首次浏览到首次购买间隔，单位：小时
first_pv_time = (
    df_sorted[df_sorted["behavior_type"] == "pv"]
    .groupby("user_id")["timestamp"]
    .min()
    .reset_index(name="first_pv_time")
)

first_buy_time = (
    df_sorted[df_sorted["behavior_type"] == "buy"]
    .groupby("user_id")["timestamp"]
    .min()
    .reset_index(name="first_buy_time")
)

buy_interval = first_pv_time.merge(
    first_buy_time,
    on="user_id",
    how="left"
)

buy_interval["first_buy_interval"] = (
    buy_interval["first_buy_time"] - buy_interval["first_pv_time"]
).dt.total_seconds() / 3600

user_feature = user_feature.merge(
    buy_interval[["user_id", "first_buy_interval"]],
    on="user_id",
    how="left"
)


# avg_buy_interval：
# 相邻购买时间差平均值，单位：小时
buy_time = df_sorted[df_sorted["behavior_type"] == "buy"][
    ["user_id", "timestamp"]
].copy()

buy_time["prev_buy_time"] = (
    buy_time.groupby("user_id")["timestamp"]
    .shift(1)
)

buy_time["buy_interval_hour"] = (
    buy_time["timestamp"] - buy_time["prev_buy_time"]
).dt.total_seconds() / 3600

avg_buy_interval = (
    buy_time.groupby("user_id")["buy_interval_hour"]
    .mean()
    .reset_index(name="avg_buy_interval")
)

user_feature = user_feature.merge(
    avg_buy_interval,
    on="user_id",
    how="left"
)


# repeat_view_before_buy：
# 首次购买前的浏览次数，用于衡量购买前浏览深度
df_with_first_buy = df_sorted.merge(
    first_buy_time,
    on="user_id",
    how="left"
)

repeat_view_before_buy = (
    df_with_first_buy[
        (df_with_first_buy["behavior_type"] == "pv")
        & (df_with_first_buy["timestamp"] < df_with_first_buy["first_buy_time"])
    ]
    .groupby("user_id")
    .size()
    .reset_index(name="repeat_view_before_buy")
)

user_feature = user_feature.merge(
    repeat_view_before_buy,
    on="user_id",
    how="left"
)

user_feature["repeat_view_before_buy"] = (
    user_feature["repeat_view_before_buy"]
    .fillna(0)
    .astype(int)
)


# =========================
# 11. 用户犹豫度特征
# =========================

# repeat_view_without_buy：
# 重复浏览但未购买的商品数量
# 判断逻辑：同一用户-商品浏览次数 > 1，且该用户没有购买过该商品
view_item_cnt = (
    df_sorted[df_sorted["behavior_type"] == "pv"]
    .groupby(["user_id", "item_id"])
    .size()
    .reset_index(name="view_cnt")
)

buy_item_set = (
    df_sorted[df_sorted["behavior_type"] == "buy"]
    [["user_id", "item_id"]]
    .drop_duplicates()
)

view_item_cnt = view_item_cnt.merge(
    buy_item_set.assign(has_buy=1),
    on=["user_id", "item_id"],
    how="left"
)

repeat_view_without_buy = (
    view_item_cnt[
        (view_item_cnt["view_cnt"] > 1)
        & (view_item_cnt["has_buy"].isna())
    ]
    .groupby("user_id")
    .size()
    .reset_index(name="repeat_view_without_buy")
)

user_feature = user_feature.merge(
    repeat_view_without_buy,
    on="user_id",
    how="left"
)

user_feature["repeat_view_without_buy"] = (
    user_feature["repeat_view_without_buy"]
    .fillna(0)
    .astype(int)
)


# long_cart_no_buy_flag：
# 长期犹豫用户，判断标准：加购次数高于75分位数，且购买次数为0
user_feature["long_cart_no_buy_flag"] = (
    (user_feature["cart_cnt"] >= user_feature["cart_cnt"].quantile(0.75))
    & (user_feature["buy_cnt"] == 0)
)


# =========================
# 12. 类目行为迁移特征
# =========================

# category_shift_cnt：
# 类目切换次数，按用户行为时间排序后，统计相邻行为中的类目变化次数
df_sorted["prev_category"] = (
    df_sorted.groupby("user_id")["item_category"]
    .shift(1)
)

df_sorted["category_changed"] = (
    (df_sorted["item_category"] != df_sorted["prev_category"])
    & df_sorted["prev_category"].notna()
)

category_shift_cnt = (
    df_sorted.groupby("user_id")["category_changed"]
    .sum()
    .reset_index(name="category_shift_cnt")
)

user_feature = user_feature.merge(
    category_shift_cnt,
    on="user_id",
    how="left"
)

user_feature["category_shift_cnt"] = (
    user_feature["category_shift_cnt"]
    .fillna(0)
    .astype(int)
)


# dominant_category_stability：
# 主类目稳定性，定义为用户连续活跃日期中关注同一主类目的最长连续天数
user_date_category = (
    df.groupby(["user_id", "date", "item_category"])
    .size()
    .reset_index(name="category_day_behavior_cnt")
)

user_date_top_category = (
    user_date_category
    .sort_values(
        ["user_id", "date", "category_day_behavior_cnt"],
        ascending=[True, True, False]
    )
    .drop_duplicates(["user_id", "date"])
    [["user_id", "date", "item_category"]]
    .rename(columns={"item_category": "daily_top_category"})
)

user_date_top_category = user_date_top_category.sort_values(["user_id", "date"])

user_date_top_category["prev_date"] = (
    user_date_top_category.groupby("user_id")["date"]
    .shift(1)
)

user_date_top_category["prev_daily_top_category"] = (
    user_date_top_category.groupby("user_id")["daily_top_category"]
    .shift(1)
)

user_date_top_category["date_gap"] = (
    user_date_top_category["date"] - user_date_top_category["prev_date"]
).dt.days

user_date_top_category["new_category_streak"] = (
    (user_date_top_category["daily_top_category"] != user_date_top_category["prev_daily_top_category"])
    | (user_date_top_category["date_gap"] != 1)
    | (user_date_top_category["date_gap"].isna())
)

user_date_top_category["category_streak_group"] = (
    user_date_top_category.groupby("user_id")["new_category_streak"]
    .cumsum()
)

category_streak_length = (
    user_date_top_category
    .groupby(["user_id", "category_streak_group"])
    .size()
    .reset_index(name="category_streak_days")
)

dominant_category_stability = (
    category_streak_length.groupby("user_id")["category_streak_days"]
    .max()
    .reset_index(name="dominant_category_stability")
)

user_feature = user_feature.merge(
    dominant_category_stability,
    on="user_id",
    how="left"
)

# cross_category_transition_flag：
# 是否频繁跨类目浏览，判断标准：类目切换次数高于75分位数
user_feature["cross_category_transition_flag"] = (
    user_feature["category_shift_cnt"] >=
    user_feature["category_shift_cnt"].quantile(0.75)
)


# =========================
# 13. 连续性特征
# =========================

# observation_end_date：
# 观察窗口结束日期，用于计算最近活跃间隔和近期行为
observation_end_date = df["date"].max()

# 每个用户的活跃日期
user_active_dates = (
    df[["user_id", "date"]]
    .drop_duplicates()
    .sort_values(["user_id", "date"])
)

# 计算连续活跃区间
user_active_dates["prev_date"] = (
    user_active_dates.groupby("user_id")["date"]
    .shift(1)
)

user_active_dates["date_gap"] = (
    user_active_dates["date"] - user_active_dates["prev_date"]
).dt.days

user_active_dates["new_streak"] = (
    (user_active_dates["date_gap"] != 1)
    | user_active_dates["date_gap"].isna()
)

user_active_dates["streak_group"] = (
    user_active_dates.groupby("user_id")["new_streak"]
    .cumsum()
)

streak_lengths = (
    user_active_dates
    .groupby(["user_id", "streak_group"])
    .size()
    .reset_index(name="consecutive_active_days")
)

consecutive_stats = (
    streak_lengths.groupby("user_id")
    .agg(
        max_consecutive_active_days=("consecutive_active_days", "max"),
        avg_consecutive_active_days=("consecutive_active_days", "mean")
    )
    .reset_index()
)

user_feature = user_feature.merge(
    consecutive_stats,
    on="user_id",
    how="left"
)


# recent_3d_behavior_cnt：
# 最近3天行为次数，观察窗口结束日向前包含3天
recent_3d_start = pd.to_datetime(observation_end_date) - pd.Timedelta(days=2)

recent_3d_behavior = (
    df[df["date"] >= recent_3d_start]
    .groupby("user_id")
    .size()
    .reset_index(name="recent_3d_behavior_cnt")
)

user_feature = user_feature.merge(
    recent_3d_behavior,
    on="user_id",
    how="left"
)


# recent_7d_behavior_cnt：
# 最近7天行为次数，观察窗口结束日向前包含7天
recent_7d_start = pd.to_datetime(observation_end_date) - pd.Timedelta(days=6)

recent_7d_behavior = (
    df[df["date"] >= recent_7d_start]
    .groupby("user_id")
    .size()
    .reset_index(name="recent_7d_behavior_cnt")
)

user_feature = user_feature.merge(
    recent_7d_behavior,
    on="user_id",
    how="left"
)


# recent_3d_buy_cnt：
# 最近3天购买次数
recent_3d_buy = (
    df[
        (df["date"] >= recent_3d_start)
        & (df["behavior_type"] == "buy")
    ]
    .groupby("user_id")
    .size()
    .reset_index(name="recent_3d_buy_cnt")
)

user_feature = user_feature.merge(
    recent_3d_buy,
    on="user_id",
    how="left"
)


# recent_7d_buy_cnt：
# 最近7天购买次数
recent_7d_buy = (
    df[
        (df["date"] >= recent_7d_start)
        & (df["behavior_type"] == "buy")
    ]
    .groupby("user_id")
    .size()
    .reset_index(name="recent_7d_buy_cnt")
)

user_feature = user_feature.merge(
    recent_7d_buy,
    on="user_id",
    how="left"
)


recent_cols = [
    "recent_3d_behavior_cnt",
    "recent_7d_behavior_cnt",
    "recent_3d_buy_cnt",
    "recent_7d_buy_cnt"
]

user_feature[recent_cols] = (
    user_feature[recent_cols]
    .fillna(0)
    .astype(int)
)


# inactive_gap_days：
# 用户沉默时长 = 观察窗口结束日期 - 用户最近一次活跃日期
user_feature["inactive_gap_days"] = (
    pd.to_datetime(observation_end_date)
    - user_feature["last_behavior_time"].dt.normalize()
).dt.days


# active_streak_flag：
# 是否为持续活跃用户，判断标准：最大连续活跃天数高于75分位数
user_feature["active_streak_flag"] = (
    user_feature["max_consecutive_active_days"] >=
    user_feature["max_consecutive_active_days"].quantile(0.75)
)


# =========================
# 14. 用户价值标签
# =========================

# purchase_frequency：
# 购买频率 = 购买次数 / 活跃天数
user_feature["purchase_frequency"] = np.where(
    user_feature["active_days"] > 0,
    user_feature["buy_cnt"] / user_feature["active_days"],
    np.nan
)

# repurchase_flag：
# 是否复购，判断标准：重复购买商品数 > 0
user_feature["repurchase_flag"] = (
    user_feature["repeat_buy_item_cnt"] > 0
)

# active_user_flag：
# 是否为活跃用户，判断标准：活跃天数高于75分位数
user_feature["active_user_flag"] = (
    user_feature["active_days"] >=
    user_feature["active_days"].quantile(0.75)
)

# high_value_user_flag：
# 是否为高价值用户，判断标准：购买次数高于75分位数
user_feature["high_value_user_flag"] = (
    user_feature["buy_cnt"] >=
    user_feature["buy_cnt"].quantile(0.75)
)

# potential_user_flag：
# 是否为潜在转化用户，判断标准：加购+收藏次数高于75分位数，且购买次数低于或等于中位数
user_feature["potential_user_flag"] = (
    (
        user_feature["cart_cnt"] + user_feature["fav_cnt"]
        >= (user_feature["cart_cnt"] + user_feature["fav_cnt"]).quantile(0.75)
    )
    & (user_feature["buy_cnt"] <= user_feature["buy_cnt"].median())
)

# loyal_user_flag：
# 是否为忠诚用户，判断标准：购买次数高于75分位数，且活跃天数高于75分位数
user_feature["loyal_user_flag"] = (
    (user_feature["buy_cnt"] >= user_feature["buy_cnt"].quantile(0.75))
    & (user_feature["active_days"] >= user_feature["active_days"].quantile(0.75))
)


# =========================
# 15. 用户转化标签
# =========================

# high_conversion_user_flag：
# 高转化用户，判断标准：浏览到购买转化率高于75分位数
user_feature["high_conversion_user_flag"] = (
    user_feature["pv_buy_ratio"] >=
    user_feature["pv_buy_ratio"].quantile(0.75)
)

# intent_user_flag：
# 潜在购买用户，判断标准：加购+收藏次数高于75分位数，且购买次数低于或等于中位数
user_feature["intent_user_flag"] = (
    (
        user_feature["cart_cnt"] + user_feature["fav_cnt"]
        >= (user_feature["cart_cnt"] + user_feature["fav_cnt"]).quantile(0.75)
    )
    & (user_feature["buy_cnt"] <= user_feature["buy_cnt"].median())
)

# low_conversion_flag：
# 低转化用户，判断标准：浏览次数高于75分位数，但浏览购买转化率低于或等于中位数
user_feature["low_conversion_flag"] = (
    (user_feature["pv_cnt"] >= user_feature["pv_cnt"].quantile(0.75))
    & (user_feature["pv_buy_ratio"] <= user_feature["pv_buy_ratio"].median())
)

# high_pv_low_buy_flag：
# 高浏览低购买用户，判断标准：浏览次数高于75分位数，且购买次数低于或等于中位数
user_feature["high_pv_low_buy_flag"] = (
    (user_feature["pv_cnt"] >= user_feature["pv_cnt"].quantile(0.75))
    & (user_feature["buy_cnt"] <= user_feature["buy_cnt"].median())
)

# high_fav_low_buy_flag：
# 高收藏低购买用户，判断标准：收藏次数高于75分位数，且购买次数低于或等于中位数
user_feature["high_fav_low_buy_flag"] = (
    (user_feature["fav_cnt"] >= user_feature["fav_cnt"].quantile(0.75))
    & (user_feature["buy_cnt"] <= user_feature["buy_cnt"].median())
)


# impulsive_buy_flag：
# 冲动型购买用户，判断标准：存在“浏览后1小时内购买同一商品”的行为
pv_records = df_sorted[df_sorted["behavior_type"] == "pv"][
    ["user_id", "item_id", "timestamp"]
].rename(columns={"timestamp": "pv_time"})

buy_records = df_sorted[df_sorted["behavior_type"] == "buy"][
    ["user_id", "item_id", "timestamp"]
].rename(columns={"timestamp": "buy_time"})

pv_buy_match = pv_records.merge(
    buy_records,
    on=["user_id", "item_id"],
    how="inner"
)

pv_buy_match["pv_to_buy_hours"] = (
    pv_buy_match["buy_time"] - pv_buy_match["pv_time"]
).dt.total_seconds() / 3600

impulsive_buy_user = (
    pv_buy_match[
        (pv_buy_match["pv_to_buy_hours"] >= 0)
        & (pv_buy_match["pv_to_buy_hours"] <= 1)
    ][["user_id"]]
    .drop_duplicates()
)

user_feature = user_feature.merge(
    impulsive_buy_user.assign(impulsive_buy_flag=True),
    on="user_id",
    how="left"
)

user_feature["impulsive_buy_flag"] = (
    user_feature["impulsive_buy_flag"]
    .fillna(False)
)


# =========================
# 16. 生命周期特征
# =========================

# new_user_flag：
# 是否为新用户，判断标准：用户行为时间跨度低于25分位数
user_feature["new_user_flag"] = (
    user_feature["user_lifetime_days"] <=
    user_feature["user_lifetime_days"].quantile(0.25)
)

# mature_user_flag：
# 是否为成熟用户，判断标准：用户行为时间跨度高于75分位数，且活跃天数高于75分位数
user_feature["mature_user_flag"] = (
    (user_feature["user_lifetime_days"] >= user_feature["user_lifetime_days"].quantile(0.75))
    & (user_feature["active_days"] >= user_feature["active_days"].quantile(0.75))
)

# silent_user_flag：
# 是否为沉默用户，判断标准：最近活跃距观察窗口结束日较久，沉默时长高于75分位数
user_feature["silent_user_flag"] = (
    user_feature["inactive_gap_days"] >=
    user_feature["inactive_gap_days"].quantile(0.75)
)

# churn_risk_flag：
# 是否为流失风险用户，判断标准：最近7天行为次数低于25分位数，且沉默时长高于75分位数
user_feature["churn_risk_flag"] = (
    (user_feature["recent_7d_behavior_cnt"] <= user_feature["recent_7d_behavior_cnt"].quantile(0.25))
    & (user_feature["inactive_gap_days"] >= user_feature["inactive_gap_days"].quantile(0.75))
)


# user_maturity_stage：
# 用户生命周期阶段。注意：判断顺序会影响最终分类。
def classify_user_stage(row):
    if row["new_user_flag"]:
        return "new_user"
    elif row["mature_user_flag"]:
        return "mature_user"
    elif row["silent_user_flag"]:
        return "silent_user"
    elif row["churn_risk_flag"]:
        return "churn_risk_user"
    else:
        return "normal_user"


user_feature["user_maturity_stage"] = user_feature.apply(
    classify_user_stage,
    axis=1
)


# retention_score：
# 留存行为综合评分 = 活跃天数分位得分 + 最大连续活跃天数分位得分 + 最近7天行为分位得分 - 沉默时长分位得分
# 分数越高，说明用户留存倾向越强
user_feature["retention_score"] = (
    user_feature["active_days"].rank(pct=True)
    + user_feature["max_consecutive_active_days"].rank(pct=True)
    + user_feature["recent_7d_behavior_cnt"].rank(pct=True)
    - user_feature["inactive_gap_days"].rank(pct=True)
)


# =========================
# 17. 字段顺序整理
# =========================

# 按照“基础行为 → 转化 → 时间 → 偏好 → 序列/犹豫 → 价值标签 → 生命周期”的逻辑排列字段
ordered_cols = [
    # 基础行为规模特征
    "user_id",
    "total_behavior_cnt",
    "pv_cnt",
    "fav_cnt",
    "cart_cnt",
    "buy_cnt",
    "active_days",
    "behavior_type_cnt",

    # 行为深度特征
    "avg_behavior_per_day",
    "avg_behavior_per_item",
    "avg_behavior_per_category",
    "max_daily_behavior_cnt",
    "min_daily_behavior_cnt",

    # 转化率特征
    "user_buy_rate",
    "user_cart_rate",
    "user_fav_rate",
    "user_pv_rate",
    "user_intent_score",
    "cart_buy_ratio",
    "fav_buy_ratio",
    "pv_buy_ratio",
    "pv_to_cart_ratio",


    # 时间活跃特征
    "first_behavior_time",
    "last_behavior_time",
    "user_lifetime_days",
    "active_hour_cnt",
    "peak_active_hour",
    "late_night_behavior_cnt",
    "morning_behavior_cnt",
    "afternoon_behavior_cnt",
    "evening_behavior_cnt",
    "morning_ratio",
    "evening_ratio",
    "late_night_ratio",
    "weekend_behavior_cnt",
    "weekday_behavior_cnt",
    "weekend_ratio",

    # 时间粒度特征
    "active_1h_slot",
    "active_2h_slot",
    "active_3h_slot",
    "active_6h_slot",
    "peak_1h_slot",
    "peak_6h_slot",

    # 连续性特征
    "max_consecutive_active_days",
    "avg_consecutive_active_days",
    "recent_3d_behavior_cnt",
    "recent_7d_behavior_cnt",
    "recent_3d_buy_cnt",
    "recent_7d_buy_cnt",
    "inactive_gap_days",
    "active_streak_flag",

    # 商品偏好特征
    "viewed_item_cnt",
    "bought_item_cnt",
    "cart_item_cnt",
    "fav_item_cnt",
    "item_diversity",
    "repeat_view_item_cnt",
    "repeat_buy_item_cnt",
    "top_item_behavior_cnt",
    "top_item_id",
    "favorite_item_ratio",
    "repeat_buy_same_item_flag",
    "single_item_dependency_flag",

    # 类目偏好特征
    "viewed_category_cnt",
    "bought_category_cnt",
    "cart_category_cnt",
    "fav_category_cnt",
    "category_diversity",
    "top_category_id",
    "top_category_behavior_cnt",
    "top_category_behavior_ratio",
    "category_shift_cnt",
    "dominant_category_stability",
    "cross_category_transition_flag",
    "cross_category_user_flag",

    # 用户价值特征
    "purchase_frequency",
    "repurchase_flag",
    "loyal_user_flag",
    "high_value_user_flag",
    "potential_user_flag",
    "active_user_flag",

    # 转化标签与犹豫标签
    "high_conversion_user_flag",
    "intent_user_flag",
    "impulsive_buy_flag",
    "low_conversion_flag",
    "hesitation_score",
    "repeat_view_before_buy",
    "repeat_view_without_buy",
    "long_cart_no_buy_flag",
    "high_pv_low_buy_flag",
    "high_fav_low_buy_flag",

    # 生命周期特征
    "new_user_flag",
    "mature_user_flag",
    "silent_user_flag",
    "churn_risk_flag",
    "user_maturity_stage",
    "retention_score",
]

# 只保留当前实际生成的字段，避免字段名拼写变化导致报错
ordered_cols = [col for col in ordered_cols if col in user_feature.columns]
remaining_cols = [col for col in user_feature.columns if col not in ordered_cols]
user_feature = user_feature[ordered_cols + remaining_cols]


# =========================
# 18. 数据格式优化
# =========================

# 浮点字段保留4位小数，使输出表更加整洁
float_cols = user_feature.select_dtypes(include="float").columns
user_feature[float_cols] = user_feature[float_cols].round(4)


# =========================
# 19. 数据检查
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
# 20. 输出中间表
# =========================

user_feature_path = Path(
    "data/intermediate_tables/User_Feature_Table.csv"
)

user_feature.to_csv(
    user_feature_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\n用户中间表已保存至：{user_feature_path}")
print(user_feature.head())
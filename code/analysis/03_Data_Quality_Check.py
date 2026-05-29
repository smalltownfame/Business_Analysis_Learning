import pandas as pd

pd.set_option(
    "display.float_format",
    lambda x: "%.2f" % x
)

# 读取数据
raw_df_path="data/processed/Clean_User_Behavior.csv"
raw_df = pd.read_csv(raw_df_path) # 读取原始数据

user_df_path = "data/intermediate_tables/User_Feature_Table.parquet"
user_df = pd.read_parquet(user_df_path) # 读取用户中间表

item_df_path = "data/intermediate_tables/Item_Feature_Table.parquet"
item_df = pd.read_parquet(item_df_path) # 读取商品中间表

category_df_path = "data/intermediate_tables/Category_Feature_Table.parquet"
category_df = pd.read_parquet(category_df_path) # 读取类目中间表



#=================================
#  第一部分：用户中间表数据质量检查
#=================================
print("\n" + "=" * 60)
print("第一部分：用户中间表数据质量检查")
print("=" * 60)


#======检查1.用户主键唯一性======
print("\n【检查1：用户主键唯一性】")
duplicate_cnt = user_df["user_id"].duplicated().sum()
print("重复user_id数量：", duplicate_cnt)


#======检查2.用户数量一致性======
print("\n【检查2：用户数量一致性】")
raw_user_cnt = raw_df["user_id"].nunique()
feature_user_cnt = user_df["user_id"].nunique()

print("原始用户数：", raw_user_cnt)
print("中间表用户数：", feature_user_cnt)


#======检查3.用户数量一致性======
print("\n【检查3：总行为数一致性】")
raw_behavior_cnt = len(raw_df)
feature_behavior_cnt = (user_df["total_behavior_cnt"].sum())
print("原始行为数：", raw_behavior_cnt)
print("中间表汇总行为数：", feature_behavior_cnt)


# ======检查4. 四类行为次数一致性======
print("\n【检查4：四类行为次数一致性】")

behavior_check_map = {
    1: "pv_cnt",
    2: "fav_cnt",
    3: "cart_cnt",
    4: "buy_cnt"
}

for behavior_code, feature_col in behavior_check_map.items():

    raw_cnt = (raw_df["behavior_type"] == behavior_code).sum()
    feature_cnt = user_df[feature_col].sum()
    diff = raw_cnt - feature_cnt

    print(f"{feature_col}:")
    print(f"  原始数据行为次数：{raw_cnt}")
    print(f"  用户中间表汇总次数：{feature_cnt}")
    print(f"  差异：{diff}")


#======检查5.缺失值统计======
print("\n【检查5：缺失值统计】")
print(user_df.isnull().sum().sort_values(ascending=False).head(20))


#======检查6.比例字段范围======
print("\n【检查6：比例字段范围】")
ratio_cols = ["user_buy_rate","user_cart_rate","user_fav_rate","user_pv_rate"]
for col in ratio_cols:print(col,user_df[col].min(),user_df[col].max())


#======检查7.行为比例计算准确性======
print("\n【检查7：行为比例计算准确性】")

user_df["rate_sum"] = (
    user_df["user_buy_rate"]
    + user_df["user_cart_rate"]
    + user_df["user_fav_rate"]
    + user_df["user_pv_rate"]
)
user_df["rate_diff"] = abs(user_df["rate_sum"] - 1)

print(user_df["rate_diff"].describe())
print("偏差大于0.001用户数：",(user_df["rate_diff"] > 0.001).sum())


#======检查8.用户生命周期合理性======
print("\n【检查8：用户生命周期合理性】")
print(user_df["user_lifetime_days"].describe())


#======检查9.随机用户验证======
sample_user = (user_df["user_id"].sample(1).iloc[0])
print("\n【检查9：随机用户验证】")
print("抽样用户：", sample_user)

#原始数据
raw_user = raw_df[raw_df["user_id"] == sample_user]
print("原始行为数：", len(raw_user))

#用户中间表
feature_user = user_df[user_df["user_id"] == sample_user]
print("中间表行为数：",feature_user["total_behavior_cnt"].iloc[0])


#======检查10.负值检查======
print("\n【检查10：负值检查】")
numeric_cols = user_df.select_dtypes(include=["int64","float64"]).columns
negative_cnt = (user_df[numeric_cols] < 0).sum()
print(negative_cnt[negative_cnt > 0])

abnormal_fields = negative_cnt[negative_cnt > 0]
print(abnormal_fields)
print("存在负值字段数量：",len(abnormal_fields))


#======检查11.比例字段异常检查======
print("\n【检查11：比例字段异常检查】")
ratio_cols = ["user_buy_rate","user_cart_rate","user_fav_rate","user_pv_rate"]
for col in ratio_cols:
    abnormal = ((user_df[col] < 0)| (user_df[col] > 1)).sum()
    print(col, abnormal)



#=================================
#  第二部分：商品中间表数据质量检查
#=================================

print("\n" + "=" * 60)
print("第二部分：商品中间表数据质量检查")
print("=" * 60)


# ====== 检查1. 商品主键唯一性 ======
print("\n【检查1：商品主键唯一性】")

duplicate_item_cnt = item_df["item_id"].duplicated().sum()

print("重复 item_id 数量：", duplicate_item_cnt)


# ====== 检查2. 商品数量一致性 ======
print("\n【检查2：商品数量一致性】")

raw_item_cnt = raw_df["item_id"].nunique()
feature_item_cnt = item_df["item_id"].nunique()

print("原始商品数：", raw_item_cnt)
print("中间表商品数：", feature_item_cnt)
print("差异：", raw_item_cnt - feature_item_cnt)


# ====== 检查3. 总行为数一致性 ======
print("\n【检查3：总行为数一致性】")

raw_behavior_cnt = len(raw_df)
item_behavior_cnt = item_df["total_behavior_cnt"].sum()

print("原始行为数：", raw_behavior_cnt)
print("商品中间表汇总行为数：", item_behavior_cnt)
print("差异：", raw_behavior_cnt - item_behavior_cnt)


# ====== 检查4. 四类行为次数一致性 ======
print("\n【检查4：四类行为次数一致性】")

behavior_check_map = {1: "pv_cnt",2: "fav_cnt",3: "cart_cnt",4: "buy_cnt"}

for behavior_code, feature_col in behavior_check_map.items():

    raw_cnt = (
        raw_df["behavior_type"] == behavior_code
    ).sum()

    feature_cnt = item_df[feature_col].sum()

    diff = raw_cnt - feature_cnt

    print(f"{feature_col}:")
    print(f"  原始数据行为次数：{raw_cnt}")
    print(f"  商品中间表汇总次数：{feature_cnt}")
    print(f"  差异：{diff}")


# ====== 检查5. 缺失值统计 ======
print("\n【检查5：缺失值统计】")
missing_item = (item_df.isnull().sum().sort_values(ascending=False))
print(missing_item[missing_item > 0])


# ====== 检查6. 比例字段范围检查 ======
print("\n【检查6：比例字段范围检查】")

item_ratio_cols = [
    "pv_buy_ratio",
    "cart_buy_ratio",
    "fav_buy_ratio",
    "pv_cart_ratio",
    "buy_user_ratio",
    "weekend_behavior_ratio",
    "late_night_behavior_ratio"
]

for col in item_ratio_cols:
    if col in item_df.columns:
        min_value = item_df[col].min()
        max_value = item_df[col].max()

        abnormal_cnt = (
            (item_df[col] < 0)
            | (item_df[col] > 1)
        ).sum()

        print(f"{col}:")
        print(f"  最小值：{min_value}")
        print(f"  最大值：{max_value}")
        print(f"  异常记录数：{abnormal_cnt}")


# ======检查7. 非负字段检查======
print("\n【检查7：非负字段检查】")

item_non_negative_cols = [
    "total_behavior_cnt",
    "unique_user_cnt",
    "active_days",
    "pv_cnt",
    "fav_cnt",
    "cart_cnt",
    "buy_cnt",
    "avg_behavior_per_user",
    "avg_daily_behavior_cnt",
    "max_daily_behavior_cnt",
    "min_daily_behavior_cnt",
    "unique_view_user_cnt",
    "unique_buy_user_cnt",
    "item_lifetime_days",
    "weekend_behavior_cnt",
    "late_night_behavior_cnt"
]

for col in item_non_negative_cols:
    if col in item_df.columns:
        negative_cnt = (item_df[col] < 0).sum()
        print(f"{col}: 负值数量 = {negative_cnt}")


# ====== 检查8. 商品生命周期合理性 ======
print("\n【检查8：商品生命周期合理性】")

if "item_lifetime_days" in item_df.columns:
    print(item_df["item_lifetime_days"].describe())
    abnormal_lifetime_cnt = (item_df["item_lifetime_days"] < 1).sum()
    print("item_lifetime_days 小于1的商品数：", abnormal_lifetime_cnt)


# ====== 检查9. 随机商品抽样验证 ======
print("\n【检查9：随机商品抽样验证】")

sample_item = (
    item_df["item_id"]
    .sample(1)
    .iloc[0]
)

print("抽样商品：", sample_item)

raw_item = raw_df[
    raw_df["item_id"] == sample_item
]

feature_item = item_df[
    item_df["item_id"] == sample_item
]

print("原始行为数：", len(raw_item))
print("中间表行为数：", feature_item["total_behavior_cnt"].iloc[0])

print("原始浏览次数：", (raw_item["behavior_type"] == 1).sum())
print("中间表浏览次数：", feature_item["pv_cnt"].iloc[0])

print("原始加购次数：", (raw_item["behavior_type"] == 2).sum())
print("中间表加购次数：", feature_item["cart_cnt"].iloc[0])

print("原始收藏次数：", (raw_item["behavior_type"] == 3).sum())
print("中间表收藏次数：", feature_item["fav_cnt"].iloc[0])

print("原始购买次数：", (raw_item["behavior_type"] == 4).sum())
print("中间表购买次数：", feature_item["buy_cnt"].iloc[0])



#=================================
#  第三部分：类目中间表数据质量检查
#=================================

print("\n" + "=" * 60)
print("第三部分：类目中间表数据质量检查")
print("=" * 60)


# ====== 检查1. 类目主键唯一性 ======
print("\n【检查1：类目主键唯一性】")

duplicate_category_cnt = category_df["item_category"].duplicated().sum()

print("重复 item_category 数量：", duplicate_category_cnt)


# ====== 检查2. 类目数量一致性 ======
print("\n【检查2：类目数量一致性】")

raw_category_cnt = raw_df["item_category"].nunique()
feature_category_cnt = category_df["item_category"].nunique()

print("原始类目数：", raw_category_cnt)
print("类目中间表类目数：", feature_category_cnt)
print("差异：", raw_category_cnt - feature_category_cnt)


# ====== 检查3. 总行为数一致性 ======
print("\n【检查3：总行为数一致性】")

raw_behavior_cnt = len(raw_df)
category_behavior_cnt = category_df["total_behavior_cnt"].sum()

print("原始行为数：", raw_behavior_cnt)
print("类目中间表汇总行为数：", category_behavior_cnt)
print("差异：", raw_behavior_cnt - category_behavior_cnt)


# ====== 检查4. 四类行为次数一致性 ======
print("\n【检查4：四类行为次数一致性】")

behavior_check_map = {1: "pv_cnt",2: "fav_cnt",3: "cart_cnt",4: "buy_cnt"}

for behavior_code, feature_col in behavior_check_map.items():

    raw_cnt = (
        raw_df["behavior_type"] == behavior_code
    ).sum()

    feature_cnt = category_df[feature_col].sum()

    diff = raw_cnt - feature_cnt

    print(f"{feature_col}:")
    print(f"  原始数据行为次数：{raw_cnt}")
    print(f"  类目中间表汇总次数：{feature_cnt}")
    print(f"  差异：{diff}")


# ====== 检查5. 缺失值统计 ======
print("\n【检查5：缺失值统计】")

missing_category = (
    category_df.isnull()
    .sum()
    .sort_values(ascending=False)
)

print(missing_category[missing_category > 0])


# ====== 检查6. 比例字段范围检查 ======
print("\n【检查6：比例字段范围检查】")

category_ratio_cols = [
    "pv_buy_ratio",
    "cart_buy_ratio",
    "fav_buy_ratio",
    "pv_cart_ratio",
    "buy_user_ratio",
    "weekend_behavior_ratio",
    "late_night_behavior_ratio",
    "top_item_ratio"
]

for col in category_ratio_cols:
    if col in category_df.columns:
        min_value = category_df[col].min()
        max_value = category_df[col].max()

        abnormal_cnt = (
            (category_df[col] < 0)
            | (category_df[col] > 1)
        ).sum()

        print(f"{col}:")
        print(f"  最小值：{min_value}")
        print(f"  最大值：{max_value}")
        print(f"  异常记录数：{abnormal_cnt}")


# ====== 检查7. 非负字段检查 ======
print("\n【检查7：非负字段检查】")

category_non_negative_cols = [
    "total_behavior_cnt",
    "unique_user_cnt",
    "unique_item_cnt",
    "active_days",
    "pv_cnt",
    "fav_cnt",
    "cart_cnt",
    "buy_cnt",
    "avg_behavior_per_user",
    "avg_behavior_per_item",
    "avg_daily_behavior_cnt",
    "max_daily_behavior_cnt",
    "min_daily_behavior_cnt",
    "unique_view_user_cnt",
    "unique_buy_user_cnt",
    "category_lifetime_days",
    "weekend_behavior_cnt",
    "late_night_behavior_cnt",
    "top_item_behavior_cnt"
]

for col in category_non_negative_cols:
    if col in category_df.columns:
        negative_cnt = (category_df[col] < 0).sum()
        print(f"{col}: 负值数量 = {negative_cnt}")


# ====== 检查8. 类目生命周期合理性 ======
print("\n【检查8：类目生命周期合理性】")

if "category_lifetime_days" in category_df.columns:
    print(category_df["category_lifetime_days"].describe())

    abnormal_lifetime_cnt = (
        category_df["category_lifetime_days"] < 1
    ).sum()

    print("category_lifetime_days 小于1的类目数：", abnormal_lifetime_cnt)


# ====== 检查9. 头部商品集中度合理性 ======
print("\n【检查9：头部商品集中度合理性】")

if "top_item_ratio" in category_df.columns:
    print(category_df["top_item_ratio"].describe())

    abnormal_top_item_ratio_cnt = (
        (category_df["top_item_ratio"] < 0)
        | (category_df["top_item_ratio"] > 1)
    ).sum()

    print("top_item_ratio 异常类目数：", abnormal_top_item_ratio_cnt)


# ====== 检查10. 随机类目抽样验证 ======
print("\n【检查10：随机类目抽样验证】")

sample_category = (
    category_df["item_category"]
    .sample(1)
    .iloc[0]
)

print("抽样类目：", sample_category)

raw_category = raw_df[
    raw_df["item_category"] == sample_category
]

feature_category = category_df[
    category_df["item_category"] == sample_category
]

print("原始行为数：", len(raw_category))
print("中间表行为数：", feature_category["total_behavior_cnt"].iloc[0])

print("原始浏览次数：", (raw_category["behavior_type"] == 1).sum())
print("中间表浏览次数：", feature_category["pv_cnt"].iloc[0])

print("原始收藏次数：", (raw_category["behavior_type"] == 2).sum())
print("中间表收藏次数：", feature_category["fav_cnt"].iloc[0])

print("原始加购次数：", (raw_category["behavior_type"] == 3).sum())
print("中间表加购次数：", feature_category["cart_cnt"].iloc[0])

print("原始购买次数：", (raw_category["behavior_type"] == 4).sum())
print("中间表购买次数：", feature_category["buy_cnt"].iloc[0])